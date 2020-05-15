#include <iostream>
#include <iomanip>
#include <fstream>
#include <list>
#include <vector>
#include <string>
#include <functional>
#include <locale>
#include <utility>
#include <cstdlib>
#include <queue>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <functional>
#include <regex>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <omp.h>
#include <mpi.h>

#define MAX_THREAD_NUM 20
#define K 5
#define MAX_REDUCERS_NUM 128
#define FILES_TO_USE 24
#define PID_TO_USE 0
#define DEBUG_FLAG 0
#define PRINT_STAT_FLAG 1
#define MAX_DOUBLE_VAL 10000.0
#define MIN_DOUBLE_VAL 0.0001

using namespace std;
typedef pair <string, int> Record;
typedef vector<Record> RecordsVector;
typedef list<Record> RecordsList;
typedef queue<Record>  RecordsQueue;
typedef vector<int> IntVector;
typedef vector<string> StringVector;
typedef queue<string> StringQueue;
typedef unordered_map<string, long> RecordsHashMap;
typedef unordered_map<int, double> TimeStamp;

int main(int argc, char **argv)
{
    // Bookkeeping variables
    long numReaders = 0, numMappers = 0, numReducers = 0, numWriters = 0, numSenders = 0, numReceivers = 0, numFiles = 24;   
    int reader_threads_completed = 0;
    int mapper_threads_completed = 0;
    int reducer_threads_completed = 0;
    int writer_threads_completed = 0;
    int sender_thread_completed = 0;
    int receiver_thread_completed = 0;

    // Timing related
    double timing_array_max[6] = {MIN_DOUBLE_VAL, MIN_DOUBLE_VAL, MIN_DOUBLE_VAL, MIN_DOUBLE_VAL, MIN_DOUBLE_VAL, MIN_DOUBLE_VAL};
    double timing_array_min[6] = {MAX_DOUBLE_VAL, MAX_DOUBLE_VAL, MAX_DOUBLE_VAL, MAX_DOUBLE_VAL, MAX_DOUBLE_VAL, MAX_DOUBLE_VAL};
    double global_timing_array_max[6] = {MIN_DOUBLE_VAL, MIN_DOUBLE_VAL, MIN_DOUBLE_VAL, MIN_DOUBLE_VAL, MIN_DOUBLE_VAL, MIN_DOUBLE_VAL};
    double global_timing_array_min[6] = {MAX_DOUBLE_VAL, MAX_DOUBLE_VAL, MAX_DOUBLE_VAL, MAX_DOUBLE_VAL, MAX_DOUBLE_VAL, MAX_DOUBLE_VAL};
    omp_lock_t timing_lock;
    omp_init_lock(&timing_lock);

    // Statistics
    double start, end;
    long processor_local_word_count = 0;
    long processor_global_word_count = 0;
    double longest_time_taken = 0;
    int number_of_files_per_processor;
    int file_index_start = 0;
    int file_index_stop = 0;

    // Variables to handle MPI
    vector<char> local_key_buffer;
    vector<int>  local_val_buffer;
    string concatKeyStr = "";

    int total_val_list_size = 0;
    int total_key_list_size = 0;
    int local_val_list_size = 0;
    int local_key_list_size = 0;

    // Various STL structurs required   
    RecordsQueue MapperWorkQueues[MAX_THREAD_NUM];
    RecordsQueue SenderWorkQueue;
    RecordsQueue ReceiverWorkQueue;
    RecordsList SenderWorkList;
    RecordsList ReceiverWorkList;
    RecordsQueue ReducerWorkQueues[MAX_THREAD_NUM]; 
    RecordsQueue WriterWorkQueues[MAX_THREAD_NUM];
    RecordsQueue TempQueues[MAX_THREAD_NUM];

    RecordsHashMap MapperHashMaps[MAX_THREAD_NUM]; 
    StringQueue filenames;

    // Parse Commandline to fill number of different threads
    if(argc >= 2 ) numReaders    = strtol(argv[1], NULL, 10);
    if(argc >= 3 ) numMappers    = strtol(argv[2], NULL, 10);
    if(argc >= 4 ) numReducers   = strtol(argv[3], NULL, 10);
    if(argc >= 5 ) numWriters    = strtol(argv[4], NULL, 10);
    if(argc >= 6 ) numFiles      = strtol(argv[5], NULL, 10);
    numSenders    = 1;
    numReceivers  = 1;

    // Declare thread STL locks types
    omp_lock_t reader_locks[MAX_THREAD_NUM];
    omp_lock_t mapper_locks[MAX_THREAD_NUM];
    omp_lock_t reducer_locks[MAX_THREAD_NUM];
    omp_lock_t writer_locks[MAX_THREAD_NUM];
    omp_lock_t sender_lock;
    omp_lock_t receiver_lock;
    omp_lock_t filenames_lock;

    omp_lock_t processor_local_word_count_lock;
    omp_lock_t reader_threads_completed_lock;
    omp_lock_t mapper_threads_completed_lock;
    omp_lock_t sender_thread_completed_lock;
    omp_lock_t receiver_thread_completed_lock;
    omp_lock_t reducer_threads_completed_lock;
    omp_lock_t writer_threads_completed_lock;

    // Initialize locks
    omp_init_lock(&(reader_threads_completed_lock));
    omp_init_lock(&(mapper_threads_completed_lock));
    omp_init_lock(&(sender_thread_completed_lock));
    omp_init_lock(&(receiver_thread_completed_lock));
    omp_init_lock(&(reducer_threads_completed_lock));
    omp_init_lock(&(writer_threads_completed_lock));
    omp_init_lock(&(processor_local_word_count_lock));
    omp_init_lock(&(filenames_lock));
    for(int i = 0; i < MAX_THREAD_NUM; i++)
    {
            omp_init_lock(&(reader_locks[i]));
            omp_init_lock(&(mapper_locks[i]));
            omp_init_lock(&(reducer_locks[i]));
            omp_init_lock(&(writer_locks[i]));
    }

    // MPI Identification Variables
    int pid;
    int numP;
    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numP);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Get_processor_name(name, &len);    

    // Alocate file numbers to various nodes
    if (numFiles % numP == 0)
    {
        number_of_files_per_processor  = (int)(numFiles / numP); 
        file_index_start = pid * number_of_files_per_processor;
        file_index_stop =  file_index_start + number_of_files_per_processor - 1;
    }
    else
    {
        number_of_files_per_processor  = (int)(numFiles / numP) + 1 ;
        file_index_start = pid * number_of_files_per_processor;
        file_index_stop =  file_index_start + number_of_files_per_processor - 1;
        if(file_index_stop > numFiles - 1) file_index_stop = numFiles - 1;
    }

    // Fill local processor work queue with file names
    int final_i;
    string index_str;
    string filename;
    for (int i = file_index_start; i <= file_index_stop; i++)
    {
        final_i = i % FILES_TO_USE;
        index_str = to_string(final_i);
        if(index_str.length() == 1) index_str = "0" + index_str;
        filename = "InputDir/file" + index_str + ".txt"; 
        filenames.push(filename.c_str());
    }

    //MPI Barrier
    MPI_Barrier(MPI_COMM_WORLD);

    // Start Time
    start = MPI_Wtime();

    // Start threads
    omp_set_num_threads(MAX_THREAD_NUM);
    #pragma omp parallel
    {
        #pragma omp master
        {

            // Reader Threads
            for (int i = 0; i < numReaders; i++)
            {
                #pragma omp task shared(numReaders, numMappers, numReducers, numWriters, numSenders, numReceivers, \
                                        reader_threads_completed, \
                                        mapper_threads_completed, \
                                        sender_thread_completed, \
                                        receiver_thread_completed, \
                                        reducer_threads_completed, \
                                        writer_threads_completed, \
                                        MapperWorkQueues, SenderWorkQueue, ReceiverWorkQueue, ReducerWorkQueues, TempQueues, WriterWorkQueues, filenames, \
                                        SenderWorkList, ReceiverWorkList, \
                                        mapper_locks, sender_lock, receiver_lock, reducer_locks, writer_locks, filenames_lock \
                                        ) 
                {
                    double thread_start_time, thread_end_time, time_elapsed;
                    thread_start_time = MPI_Wtime();
                    int mapper_index;
                    string filename;
                    bool break_loop = false;
                    bool task_flag = false;
                    ifstream fhnd;
                    string word;
                    while(true)
                    {
                        // Check if all files have been read
                        omp_set_lock(&filenames_lock);
                        if (filenames.empty()) break_loop = true;
                        if(filenames.size() > 0)
                        {
                            filename = filenames.front();
                            filenames.pop();
                            task_flag = true;                    
                        }                        
                        omp_unset_lock(&filenames_lock);  

                        if (break_loop == true) break;
                        if(task_flag == true)
                        {
                            fhnd.open(filename.c_str());
                            while (fhnd >> word)
                            {
                                if(word.length() > 0)
                                {
                                    mapper_index = rand() % numMappers;
                                    omp_set_lock(&(mapper_locks[mapper_index]));
                                    MapperWorkQueues[mapper_index].push(make_pair(word, 1));
                                    omp_unset_lock(&(mapper_locks[mapper_index]));                                    
                                }
                            }
                            fhnd.close();

                        }
                        task_flag = false;
                    }
                    omp_set_lock(&(reader_threads_completed_lock));
                    reader_threads_completed++;
                    omp_unset_lock(&(reader_threads_completed_lock));

                    // Timing calculations
                    thread_end_time = MPI_Wtime();
                    time_elapsed = thread_end_time - thread_start_time;
                    omp_set_lock(&timing_lock);
                    if(timing_array_max[0] < time_elapsed && time_elapsed < MAX_DOUBLE_VAL) timing_array_max[0] = time_elapsed;
                    if(time_elapsed < timing_array_min[0] && time_elapsed > MIN_DOUBLE_VAL) timing_array_min[0] = time_elapsed;
                    omp_unset_lock(&timing_lock);


                }
            }   

            // Mapper Threads
            for (int i = 0; i < numMappers; i++)
            {
                #pragma omp task shared(numReaders, numMappers, numReducers, numWriters, numSenders, numReceivers, \
                                        reader_threads_completed, \
                                        mapper_threads_completed, \
                                        sender_thread_completed, \
                                        receiver_thread_completed, \
                                        reducer_threads_completed, \
                                        writer_threads_completed, \
                                        MapperWorkQueues, SenderWorkQueue, ReceiverWorkQueue, ReducerWorkQueues, TempQueues, WriterWorkQueues, filenames, \
                                        SenderWorkList, ReceiverWorkList, \
                                        mapper_locks, sender_lock, receiver_lock, reducer_locks, writer_locks \
                                        ) 
                {

                    Record record;
                    bool break_loop = false;
                    bool task_flag = false;
                    int hash_val = 0;
                    int cur_reader_threads_completed = 0;
                    RecordsHashMap HashMap;
                    bool first_flag = true;
                    double thread_start_time, thread_end_time, time_elapsed;

                    while(true)
                    {                 
                        omp_set_lock(&(reader_threads_completed_lock));
                        cur_reader_threads_completed = reader_threads_completed;
                        omp_unset_lock(&(reader_threads_completed_lock));

                        omp_set_lock(&(mapper_locks[i]));
                        if (MapperWorkQueues[i].size() == 0 && cur_reader_threads_completed  == numReaders) break_loop = true;
                        if(MapperWorkQueues[i].size() > 0)
                        {
                            if(first_flag == true)
                            {
                                first_flag = false;
                                thread_start_time = MPI_Wtime();
                            }
                            record = MapperWorkQueues[i].front();
                            MapperWorkQueues[i].pop();
                            task_flag = true;
                            
                        }                        
                        omp_unset_lock(&(mapper_locks[i]));

                        if (break_loop == true) break;
                        if(task_flag == true)
                        {
                            // Use hash table to maintain thread word count
                            auto it = HashMap.find(record.first);
                            if(it != HashMap.end()) 
                                it->second = it->second + record.second;
                            else
                                HashMap[record.first] = record.second;

                            // Update processor local word count
                            omp_set_lock(&(processor_local_word_count_lock));
                            processor_local_word_count++;
                            omp_unset_lock(&(processor_local_word_count_lock)); 

                        }
                        task_flag = false;        
                    }

                    // Add to sender list
                    RecordsHashMap::iterator it = HashMap.begin();
                    while(it != HashMap.end())
                    {
                        omp_set_lock(&sender_lock);
                        SenderWorkList.push_back(make_pair(it->first, it->second));
                        omp_unset_lock(&sender_lock);
                        it++;
                    }

                    // Update mapper threads complted count
                    omp_set_lock(&(mapper_threads_completed_lock));
                    mapper_threads_completed++;
                    omp_unset_lock(&(mapper_threads_completed_lock));


                    // Timing calculations
                    thread_end_time = MPI_Wtime();
                    time_elapsed = thread_end_time - thread_start_time;
                    omp_set_lock(&timing_lock);
                    if(timing_array_max[1] < time_elapsed && time_elapsed < MAX_DOUBLE_VAL) timing_array_max[1] = time_elapsed;
                    if(time_elapsed < timing_array_min[1] && time_elapsed > MIN_DOUBLE_VAL) timing_array_min[1] = time_elapsed;
                    omp_unset_lock(&timing_lock);


                }
            }

            // Sender Thread
            #pragma omp task shared(numReaders, numMappers, numReducers, numWriters, numSenders, numReceivers, \
                                    reader_threads_completed, \
                                    mapper_threads_completed, \
                                    sender_thread_completed, \
                                    receiver_thread_completed, \
                                    reducer_threads_completed, \
                                    writer_threads_completed, \
                                    MapperWorkQueues, SenderWorkQueue, ReceiverWorkQueue, ReducerWorkQueues, TempQueues, WriterWorkQueues, filenames, \
                                    SenderWorkList, ReceiverWorkList, \
                                    mapper_locks, sender_lock, receiver_lock, reducer_locks, writer_locks \
                                    ) 
            {
                Record record;
                bool break_loop = false;
                bool task_flag = false;
                int cur_mapper_threads_completed = 0;  
                double thread_start_time, thread_end_time, time_elapsed;       
                
                while(true) 
                {
                    omp_set_lock(&(mapper_threads_completed_lock));
                    cur_mapper_threads_completed = mapper_threads_completed;
                    omp_unset_lock(&(mapper_threads_completed_lock));                    

                    if (cur_mapper_threads_completed  == numMappers) break;
                    usleep(100 * 1000);
                }

                thread_start_time = MPI_Wtime();

                // SenderWorkList is what needs to be sent to the receivers of all processes
                for(auto const & record : SenderWorkList)
                {
                    concatKeyStr += record.first + " ";
                    local_val_buffer.push_back(record.second);
                }

                local_key_list_size = (int) concatKeyStr.size();
                local_val_list_size  = (int) local_val_buffer.size();

                // Allocate local buffers
                copy(concatKeyStr.begin(), concatKeyStr.end(), std::back_inserter(local_key_buffer));

                if(DEBUG_FLAG)
                {
                    cout<<"Local key list size of process "<<pid<<" : "<<local_key_list_size<<"\n";
                    cout<<"Local key buffer size of process "<<pid<<" : "<<local_key_buffer.size()<<"\n";

                    cout<<"Local val list size of process "<<pid<<" : "<<local_val_list_size<<"\n";
                    cout<<"Local val buffer size of process "<<pid<<" : "<<local_val_buffer.size()<<"\n";
                }

                // Send Entire Vector to Receiver Here
                omp_set_lock(&(receiver_lock));
                ReceiverWorkList.splice(ReceiverWorkList.end(), SenderWorkList);
                omp_unset_lock(&(receiver_lock));

                // Set sender thread flag to complete
                omp_set_lock(&(sender_thread_completed_lock));
                sender_thread_completed++;
                omp_unset_lock(&(sender_thread_completed_lock));

                // Timing calculations
                thread_end_time = MPI_Wtime();
                time_elapsed = thread_end_time - thread_start_time;
                omp_set_lock(&timing_lock);
                if(timing_array_max[2] < time_elapsed && time_elapsed < MAX_DOUBLE_VAL) timing_array_max[2] = time_elapsed;
                if(time_elapsed < timing_array_min[2] && time_elapsed > MIN_DOUBLE_VAL) timing_array_min[2] = time_elapsed;
                omp_unset_lock(&timing_lock);
            }

            // Receiver Thread Here
            #pragma omp task shared(numReaders, numMappers, numReducers, numWriters, numSenders, numReceivers, \
                                    reader_threads_completed, \
                                    mapper_threads_completed, \
                                    sender_thread_completed, \
                                    receiver_thread_completed, \
                                    reducer_threads_completed, \
                                    writer_threads_completed, \
                                    MapperWorkQueues, SenderWorkQueue, ReceiverWorkQueue, ReducerWorkQueues, TempQueues, WriterWorkQueues, filenames, \
                                    SenderWorkList, ReceiverWorkList, \
                                    mapper_locks, sender_lock, receiver_lock, reducer_locks, writer_locks \
                                    ) 
            {
                size_t hash_val = 0;
                size_t processIndex, reducerIndex;
                int cur_sender_thread_completed = 0;
                double thread_start_time, thread_end_time, time_elapsed;                         
                while(true) 
                {
                    omp_set_lock(&(sender_thread_completed_lock));
                    cur_sender_thread_completed = sender_thread_completed;
                    omp_unset_lock(&(sender_thread_completed_lock)); 
                    if (cur_sender_thread_completed  == numSenders) break;
                    usleep(100 * 1000);
                }


                thread_start_time = MPI_Wtime();

                // Allocate count and displs arrays
                vector<int> local_key_counts(numP, 0);
                vector<int> local_key_displs(numP, 0);
                vector<int> local_val_counts(numP, 0);
                vector<int> local_val_displs(numP, 0);

                // Gather all lengths from all processes               
                MPI_Allgather(&local_key_list_size,  1, MPI_INT, local_key_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);                
                MPI_Allgather(&local_val_list_size,  1, MPI_INT, local_val_counts.data(), 1, MPI_INT, MPI_COMM_WORLD); 

                if(DEBUG_FLAG && pid == PID_TO_USE)
                {
                    cout<<"\nDEBUG STUFF from PID: "<<pid<<"\n";
                    for(int ind = 0; ind < numP; ind++)
                    {
                        cout<<"Local Key Counts: "<<local_key_counts[ind]<<"\n";
                        cout<<"Local Val Counts: "<<local_val_counts[ind]<<"\n";
                    }
                    cout<<"DEBUG END\n\n";
                }



                // Reduce length of char sub arrays
                MPI_Allreduce(&local_key_list_size, &total_key_list_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&local_val_list_size,  &total_val_list_size,  1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                if(DEBUG_FLAG && pid == PID_TO_USE)
                {
                    cout<<"Global Statictics from PID : "<<pid<<"\n";
                    cout<<"Global key buffer size: "<<total_key_list_size<<"\n";
                    cout<<"Global val buffer size: "<<total_val_list_size<<"\n\n";
                }
                
                // Allocate arrays
                vector<char> global_key_buffer(total_key_list_size, 0);
                vector<int>  global_val_buffer(total_val_list_size, 0);

                // Calculate displs
                local_key_displs[0] = 0;
                local_val_displs[0] = 0;                
                for(int ind = 1; ind < numP; ind++)
                {
                    local_key_displs[ind] = local_key_displs[ind - 1] + local_key_counts[ind - 1];
                    local_val_displs[ind] = local_val_displs[ind - 1] + local_val_counts[ind - 1];
                }

                // All Gatherv
                MPI_Allgatherv(local_key_buffer.data(), local_key_list_size, MPI_CHAR, global_key_buffer.data(), 
                               local_key_counts.data(), local_key_displs.data(), MPI_CHAR, MPI_COMM_WORLD);
                MPI_Allgatherv(local_val_buffer.data(), local_val_list_size, MPI_INT,  global_val_buffer.data(), 
                               local_val_counts.data(), local_val_displs.data(), MPI_INT, MPI_COMM_WORLD);



                //Parse key string
                string key_str(global_key_buffer.begin(), global_key_buffer.end());
                istringstream iss(key_str);
                vector<string> global_key_vector(istream_iterator<string>{iss}, istream_iterator<string>());
                
                //cout<<"Here0 pid: "<<pid<<endl;

                // Debug
                if(DEBUG_FLAG && pid == PID_TO_USE)
                {
                    cout<<"Parsing Statictics from PID : "<<pid<<"\n";
                    cout<<"Key Str size: "<<key_str.size()<<endl;
                    cout<<"Global key vector  size: "<<global_key_vector.size()<<endl;
                    cout<<"Global val buffer size: "<<global_val_buffer.size()<<"\n\n";
                }

                // Iterate and hash to appropriate reducer queue
                auto iter_key = global_key_vector.begin();
                auto iter_val = global_val_buffer.begin();
                while(iter_key != global_key_vector.end() || iter_val != global_val_buffer.end())
                {
                    hash_val = hash<string>{}(*iter_key);
                    processIndex = hash_val % numP;

                    // If in the right process
                    if(processIndex == pid)
                    {
                        // Hash to right reducer
                        reducerIndex   = hash_val % numReducers;
                        omp_set_lock(&(reducer_locks[reducerIndex]));
                        ReducerWorkQueues[reducerIndex].push(make_pair(*iter_key, *iter_val));
                        omp_unset_lock(&(reducer_locks[reducerIndex]));
                    }

                    iter_key++;
                    iter_val++;
                }

                MPI_Barrier(MPI_COMM_WORLD);
                //cout<<"Here "<<pid<<endl;

                omp_set_lock(&(receiver_thread_completed_lock));
                receiver_thread_completed++;
                omp_unset_lock(&(receiver_thread_completed_lock));

                // Timing calculations
                thread_end_time = MPI_Wtime();
                time_elapsed = thread_end_time - thread_start_time;
                omp_set_lock(&timing_lock);
                if(timing_array_max[3] < time_elapsed && time_elapsed < MAX_DOUBLE_VAL) timing_array_max[3] = time_elapsed;
                if(time_elapsed < timing_array_min[3] && time_elapsed > MIN_DOUBLE_VAL) timing_array_min[3] = time_elapsed;
                omp_unset_lock(&timing_lock);

            }

            // Reducer Threads
            for (int i = 0; i < numReducers; i++)
            {
                #pragma omp task shared(numReaders, numMappers, numReducers, numWriters, numSenders, numReceivers, \
                                        reader_threads_completed, \
                                        mapper_threads_completed, \
                                        sender_thread_completed, \
                                        receiver_thread_completed, \
                                        reducer_threads_completed, \
                                        writer_threads_completed, \
                                        MapperWorkQueues, SenderWorkQueue, ReceiverWorkQueue, ReducerWorkQueues, TempQueues, WriterWorkQueues, filenames, \
                                        SenderWorkList, ReceiverWorkList, \
                                        mapper_locks, sender_lock, receiver_lock, reducer_locks, writer_locks \
                                        ) 
                {
                    Record record;
                    bool break_loop = false;
                    bool task_flag = false;
                    int write_val = 0;
                    int cur_receiver_thread_completed = 0;
                    bool first_flag = true;
                    RecordsHashMap HashMap;
                    double thread_start_time, thread_end_time, time_elapsed;
                    while(true)
                    {                 
                        omp_set_lock(&(receiver_thread_completed_lock));
                        cur_receiver_thread_completed  = receiver_thread_completed ;
                        omp_unset_lock(&(receiver_thread_completed_lock));

                        omp_set_lock(&(reducer_locks[i]));
                        if (ReducerWorkQueues[i].size() == 0 && cur_receiver_thread_completed  == numReceivers) break_loop = true;
                        if(ReducerWorkQueues[i].size() > 0)
                        {
                            if(first_flag == true)
                            {
                                first_flag = false;
                                # pragma omp critical
                                thread_start_time = MPI_Wtime();
                            }
                            record = ReducerWorkQueues[i].front();
                            ReducerWorkQueues[i].pop();
                            task_flag = true;                            
                        }
                        omp_unset_lock(&(reducer_locks[i]));
                        
                        if (break_loop == true) break;
                        if(task_flag == true)
                        {
                            auto it = HashMap.find(record.first);
                            if(it != HashMap.end()) 
                                it->second = it->second + record.second;
                            else
                                HashMap[record.first] = record.second;
                        }
                        task_flag = false;
                    }

                    RecordsHashMap::iterator it = HashMap.begin();
                    while(it != HashMap.end())
                    {
                        write_val = rand() % numWriters;
                        omp_set_lock(&(writer_locks[write_val]));
                        WriterWorkQueues[write_val].push(make_pair(it->first, it->second));
                        omp_unset_lock(&(writer_locks[write_val]));
                        ++it;
                    }
                    omp_set_lock(&(reducer_threads_completed_lock));
                    reducer_threads_completed++;
                    omp_unset_lock(&(reducer_threads_completed_lock));
                
                    // Timing calculations
                    thread_end_time = MPI_Wtime();
                    time_elapsed = thread_end_time - thread_start_time;
                    omp_set_lock(&timing_lock);
                    if(timing_array_max[4] < time_elapsed && time_elapsed < MAX_DOUBLE_VAL) timing_array_max[4] = time_elapsed;
                    if(time_elapsed < timing_array_min[4] && time_elapsed > MIN_DOUBLE_VAL) timing_array_min[4] = time_elapsed;
                    omp_unset_lock(&timing_lock);
                }  
            }

            // Write word counts to files
            for (int i = 0; i < numWriters; i++)
            {
                #pragma omp task shared(numReaders, numMappers, numReducers, numWriters, numSenders, numReceivers, \
                                        reader_threads_completed, \
                                        mapper_threads_completed, \
                                        sender_thread_completed, \
                                        receiver_thread_completed, \
                                        reducer_threads_completed, \
                                        writer_threads_completed, \
                                        MapperWorkQueues, ReceiverWorkQueue, ReducerWorkQueues, TempQueues, WriterWorkQueues, filenames, \
                                        mapper_locks, sender_lock, receiver_lock, reducer_locks, writer_locks \
                                        ) 
                {
                    double thread_start_time, thread_end_time, time_elapsed;
                    string index_str = to_string(pid * numWriters + i);
                    string filename = "OutputDir/" + index_str + ".txt";

                    while(reducer_threads_completed < numReducers) 
                    {
                        usleep(100 * 1000);
                    }
                    thread_start_time = MPI_Wtime();

                    ofstream fout;
                    fout.open(filename.c_str());
                    Record record;
                    while(!WriterWorkQueues[i].empty())
                    {
                        record = WriterWorkQueues[i].front();
                        fout<<record.first<<", "<<record.second<<endl;
                        WriterWorkQueues[i].pop();
                    } 
                    fout.close();

                    // Timing calculations
                    thread_end_time = MPI_Wtime();
                    time_elapsed = thread_end_time - thread_start_time;
                    omp_set_lock(&timing_lock);
                    if(timing_array_max[5] < time_elapsed && time_elapsed < MAX_DOUBLE_VAL) timing_array_max[5] = time_elapsed;
                    if(time_elapsed < timing_array_min[5] && time_elapsed > MIN_DOUBLE_VAL) timing_array_min[5] = time_elapsed;
                    omp_unset_lock(&timing_lock);
                }
            }

        #pragma omp taskwait
        }
    }

    // Calculate end time
    end = MPI_Wtime();
    double total_time_taken = (end - start);

    // Gather total word counts
    MPI_Reduce(&processor_local_word_count, &processor_global_word_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time_taken, &longest_time_taken, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 

    // Gather global timing statistics
    for(int ind = 0; ind < 6; ind++)
    {
        MPI_Reduce(&(timing_array_max[ind]), &(global_timing_array_max[ind]), 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 
        MPI_Reduce(&(timing_array_min[ind]), &(global_timing_array_min[ind]), 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD); 
    }
 
    // TIMING Calculations
    if(PRINT_STAT_FLAG && pid == PID_TO_USE)
    {
        cout<<"\nFINAL STATISTICS:\n";
        cout<<"Number of Processors    : "<<numP<<"\n";
        cout<<"Number of Readers       : "<<numReaders<<"\n";
        cout<<"Number of Mappers       : "<<numMappers<<"\n";
        cout<<"Number of Reducers      : "<<numReducers<<"\n";
        cout<<"Number of Writers       : "<<numWriters<<"\n";
        cout<<"Number of Files         : "<<numFiles<<"\n\n";
        cout<<"Global Total Word Count : "<<processor_global_word_count<<"\n";
        cout<<"Longest Total Time      : "<<longest_time_taken<<"\n\n";
        cout<<"Longest Reader Time     : "<< global_timing_array_max[0]<<"\n";
        cout<<"Shortest Reader Time    : "<< global_timing_array_min[0]<<"\n";
        cout<<"Longest Mapper Time     : "<< global_timing_array_max[1]<<"\n";
        cout<<"Shortest Mapper Time    : "<< global_timing_array_min[1]<<"\n";
        cout<<"Longest Sender Time     : "<< global_timing_array_max[2]<<"\n";
        cout<<"Shortest Sender Time    : "<< global_timing_array_min[2]<<"\n";
        cout<<"Longest Receiver Time   : "<< global_timing_array_max[3]<<"\n";
        cout<<"Shortest Receiver Time  : "<< global_timing_array_min[3]<<"\n";
        cout<<"Longest Reducer Time    : "<< global_timing_array_max[4]<<"\n";
        cout<<"Shortest Reducer Time   : "<< global_timing_array_min[4]<<"\n";
        cout<<"Longest Writer Time     : "<< global_timing_array_max[5]<<"\n";
        cout<<"Shortest Writer Time    : "<< global_timing_array_min[5]<<"\n\n";
    }

    // Finalize MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

