#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <queue>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <functional>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#define MAX_THREAD_NUM 20
#define K 5
#define MAX_REDUCERS_NUM 128
#define FILES_TO_USE 10

using namespace std;

typedef pair <string, int> Record;
typedef vector<Record> RecordsVector;
typedef queue<Record>  RecordsQueue;
typedef vector<int> IntVector;
typedef vector<string> StringVector;
typedef queue<string> StringQueue;
typedef unordered_map<string, long> RecordsHashMap;



//function for getting vector from each file with each element as <word,1>
RecordsVector readRecordsFromFile(string filename)
{
	RecordsVector records;
	ifstream fhnd;
	fhnd.open(filename.c_str());

	for(string line; getline(fhnd, line);)
	{
		if(line.empty()) continue;
		istringstream iss(line);
		do
		{
			string word;
        	iss >> word;
        	if(word.find_first_not_of(' ') != string::npos) records.push_back(make_pair(word, 1));
    	} while (iss);
	}
	fhnd.close();
	return records;
}

unsigned long hash_str(const char * str)
{
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

int main(int argc, char **argv)
{
    long numReaders = 0, numMappers = 0, numReducers = 0, numWriters = 0;
    int cur_read_task_num   = 0;
    int cur_map_task_num    = 0;
    int cur_reduce_task_num = 0;
    int cur_write_task_num  = 0;    
    int reader_threads_completed = 0;
    int mapper_threads_completed = 0;
    int reducer_threads_completed = 0;
    int writer_threads_completed = 0;
    double start, end;
    long global_word_count = 0;
		
	RecordsQueue MapperWorkQueues[MAX_THREAD_NUM], TempQueues[MAX_THREAD_NUM];
	RecordsQueue ReducerWorkQueues[MAX_THREAD_NUM]; 
    RecordsQueue WriterWorkQueues[MAX_THREAD_NUM]; 

	StringQueue filenames;
    RecordsHashMap GlobalHashMaps[MAX_THREAD_NUM];

    //Input from user about the number of threads for each type
    numReaders  = strtol(argv[1], NULL, 10);
    numMappers  = strtol(argv[2], NULL, 10);
    numReducers = strtol(argv[3], NULL, 10);
    numWriters  = strtol(argv[4], NULL, 10);

    //Decalre all the different types of OpneMP locks (It is a variable of type lock.
    omp_lock_t mapper_locks[MAX_THREAD_NUM];
    omp_lock_t reducer_locks[MAX_THREAD_NUM];
    omp_lock_t writer_locks[MAX_THREAD_NUM];

    omp_lock_t global_word_count_lock;
    //These are to make sure the threads one level before are done executing fully before next sets of threads start
    omp_lock_t reader_threads_completed_lock;
    omp_lock_t mapper_threads_completed_lock;
    omp_lock_t reducer_threads_completed_lock;
    omp_lock_t writer_threads_completed_lock;

    //initialze routine for different OpenMP locks
    omp_init_lock(&(reader_threads_completed_lock));
    omp_init_lock(&(mapper_threads_completed_lock));
    omp_init_lock(&(reducer_threads_completed_lock));
    omp_init_lock(&(writer_threads_completed_lock));
    omp_init_lock(&(global_word_count_lock));
    
    //Initialize routine for each type of thread * no of threads
    for(int i = 0; i < MAX_THREAD_NUM; i++)
    {
            omp_init_lock(&(mapper_locks[i]));
            omp_init_lock(&(reducer_locks[i]));
            omp_init_lock(&(writer_locks[i]));
    }


    //get file names
    for (int i = 1; i <= FILES_TO_USE; i++)
    {
    	ostringstream oss;
		oss <<"InputDir/"<<i<<".txt";
  		filenames.push(oss.str());
    }

    //set max number of threads
    omp_set_num_threads(MAX_THREAD_NUM);
    
    //start the parallel code with different threads
    #pragma omp parallel
    {
        //Specifies the beginning of a code block that must be executed only once by the master thread.
    	#pragma omp master
    	{
    		start = omp_get_wtime();
    		cout<<"ReaderThreads:" <<numReaders <<endl;
    		cout<<"MapperThreads:" <<numMappers <<endl;
    		cout<<"ReducerThreads:"<<numReducers<<endl;
    		cout<<"WriterThreads:" <<numWriters<<endl;


            //Reader Block
    		for (int i = 0; i < numReaders; i++)
    		{
                //All references to a list item within a task refer to the storage area of the original variable
    			#pragma omp task shared(numReaders, numMappers, numReducers, numWriters, \
                                        reader_threads_completed, \
    									mapper_threads_completed, \
    									reducer_threads_completed, \
    									writer_threads_completed, \
    									MapperWorkQueues, ReducerWorkQueues, TempQueues, WriterWorkQueues, filenames, \
                                        mapper_locks, reducer_locks, writer_locks \
    									) 
    			{
                    int mapper_index;
                    
                    //loop over the file name till empty
    				while(!filenames.empty())
    				{
    					string filename;
                        // executed just by single thread at a time
    					#pragma omp critical
                        {
    	                    filename = filenames.front();
	    					filenames.pop();
	    				}

	    				if(filename.size() != 0)
	    				{
                            //records is of type RecordsVector (a vector) with each element as a pair.
                            //There is a record vector for each file
    						RecordsVector records = readRecordsFromFile(filename);
    						for(int j = 0; j  < records.size(); j++)
                                
                            //Loop over all the elements of the vector "record" for that specific file
    						{
                                //generate mapper index for each element
                                mapper_index = rand() % numMappers;
                                //Start lock for each elemenst index to avoid conflicts in variable in memory for temp and mapper work queues
                                omp_set_lock(&(mapper_locks[mapper_index]));
                                MapperWorkQueues[mapper_index].push(records[j]);
                                TempQueues[mapper_index].push(records[j]);
                                omp_unset_lock(&(mapper_locks[mapper_index]));
    						}
    						cout<<filename<<" "<< i<<endl;
    					}
    				}
                    //Making sure the total threads are not conflicted
                    omp_set_lock(&(reader_threads_completed_lock));
                    reader_threads_completed++;
                    omp_unset_lock(&(reader_threads_completed_lock));

    			}
    		}	

            //Mapper Block
            for (int i = 0; i < numMappers; i++)
            {
                #pragma omp task shared(numReaders, numMappers, numReducers, numWriters, \
                                        reader_threads_completed, \
                                        mapper_threads_completed, \
                                        reducer_threads_completed, \
                                        writer_threads_completed, \
                                        MapperWorkQueues, ReducerWorkQueues, TempQueues, WriterWorkQueues, filenames, \
                                        mapper_locks, reducer_locks, writer_locks \
                                        ) 
                {

                    Record record;
                    bool break_loop = false, reduce_flag = false;
                    int size = 0;
                    int hash_val = 0;
                    int cur_reader_threads_completed = 0;
                    //Making it run till it breaks inside the while loop
                    while(1 == 1)
                    {                 
                        omp_set_lock(&(reader_threads_completed_lock));
                        cur_reader_threads_completed = reader_threads_completed;
                        omp_unset_lock(&(reader_threads_completed_lock));

                        
                        omp_set_lock(&(mapper_locks[i]));
                        //Break loop if the mapper work queue is emoty and teh reader threads are all done.
                        if (MapperWorkQueues[i].size() == 0 && cur_reader_threads_completed  == numReaders) break_loop = true;
                        if(MapperWorkQueues[i].size() > 0)
                        {
                            record = MapperWorkQueues[i].front();
                            MapperWorkQueues[i].pop();
                            reduce_flag = true;
                            
                        }                        
                        omp_unset_lock(&(mapper_locks[i]));

                        if (break_loop == true) break;
                        //If getting the record was successful, apply hash
                        if(reduce_flag == true)
                        {
                            hash_val = hash_str(record.first.c_str()) % numReducers;

                            omp_set_lock(&(reducer_locks[hash_val]));
                            ReducerWorkQueues[hash_val].push(record);
                            omp_unset_lock(&(reducer_locks[hash_val]));
                            
                            omp_set_lock(&(global_word_count_lock));
                            global_word_count++;
                            omp_unset_lock(&(global_word_count_lock)); 

                        }
                        reduce_flag = false;        
                    }
                    omp_set_lock(&(mapper_threads_completed_lock));
                    mapper_threads_completed++;
                    omp_unset_lock(&(mapper_threads_completed_lock));
                }
            }

            //reducer block
            for (int i = 0; i < numReducers; i++)
            {
                #pragma omp task shared(numReaders, numMappers, numReducers, numWriters, \
                                        reader_threads_completed, \
                                        mapper_threads_completed, \
                                        reducer_threads_completed, \
                                        writer_threads_completed, \
                                        MapperWorkQueues, ReducerWorkQueues, TempQueues, WriterWorkQueues, filenames, \
                                        mapper_locks, reducer_locks, writer_locks \
                                        ) 
                {
                    Record record;
                    bool break_loop = false, hash_flag = false;
                    int size = 0;
                    int write_val = 0;
                    int cur_mapper_threads_completed = 0;
                    RecordsHashMap HashMap;
                    while(1 == 1)
                    {                 
                        omp_set_lock(&(mapper_threads_completed_lock));
                        cur_mapper_threads_completed = mapper_threads_completed;
                        omp_unset_lock(&(mapper_threads_completed_lock));

                        omp_set_lock(&(reducer_locks[i]));
                        //If all mapper are done writing to the reducer queue accoridng to their hash value
                        if (ReducerWorkQueues[i].size() == 0 && cur_mapper_threads_completed  == numMappers) break_loop = true;
                        if(ReducerWorkQueues[i].size() > 0)
                        {
                            record = ReducerWorkQueues[i].front();
                            ReducerWorkQueues[i].pop();
                            hash_flag = true;                            
                        }
                        omp_unset_lock(&(reducer_locks[i]));
                        
                        if (break_loop == true) break;
                        if(hash_flag == true)
                        //increse value for each word type if same word and put in hash map
                        {
                            auto it = HashMap.find(record.first);
                            if(it != HashMap.end()) 
                                it->second = it->second + 1;
                            else
                                HashMap[record.first] = 1;
                        }
                        hash_flag = false;
                    }

                    RecordsHashMap::iterator it = HashMap.begin();
                    //Write to work queue
                    while(it != HashMap.end())
                    {
                        write_val = rand() % numWriters;
                        omp_set_lock(&(writer_locks[write_val]));
                        WriterWorkQueues[write_val].push(make_pair(it->first, it->second));
                        omp_unset_lock(&(writer_locks[write_val]));
                        it++;
                    }



                    omp_set_lock(&(reducer_threads_completed_lock));
                    reducer_threads_completed++;
                    omp_unset_lock(&(reducer_threads_completed_lock));
                }  
            }
            //For Writer
            for (int i = 0; i < numWriters; i++)
            {
                #pragma omp task shared(numReaders, numMappers, numReducers, numWriters, \
                                        reader_threads_completed, \
                                        mapper_threads_completed, \
                                        reducer_threads_completed, \
                                        writer_threads_completed, \
                                        MapperWorkQueues, ReducerWorkQueues, TempQueues, WriterWorkQueues, filenames, \
                                        mapper_locks, reducer_locks, writer_locks \
                                        ) 
                {
                    while(reducer_threads_completed < numReducers) 
                    {
                        usleep(500 * 1000);
                    }

                    ostringstream oss;
                    oss <<"OutputDir/"<<i+1<<".txt";
                    ofstream fout;
                    fout.open(oss.str());
                    Record record;
                    while(!WriterWorkQueues[i].empty())
                    {
                        record = WriterWorkQueues[i].front();
                        fout<<record.first<<", "<<record.second<<endl;
                        WriterWorkQueues[i].pop();
                    } 
                    fout.close();
                }

            }





    	#pragma omp taskwait
    	}
    }

	end = omp_get_wtime();
    cout<<"Time Taken:"<<end - start<<" secs"<<endl;
    long mapper_sum_count = 0;
    long reducer_sum_count = 0;
    for (int i = 0; i < numMappers; i++)
    {
        cout<<"Mapper "<<i<<" contains "<<MapperWorkQueues[i].size()<<" words, Temp "<<i<<" contains "<<TempQueues[i].size()<<" words\n";
        mapper_sum_count += TempQueues[i].size();
    }
    cout<<endl;
    for (int i = 0; i < numReducers; i++)
    { 
        cout<<"Reducer "<<i<<" contains "<<ReducerWorkQueues[i].size()<<" words\n";
        reducer_sum_count += ReducerWorkQueues[i].size();
    }
    cout<<endl;

    cout<<"Global Word Count: "<<global_word_count<<endl;
    cout<<"Mapper Total Count: "<<mapper_sum_count<<endl;
    cout<<"Reducer Total Count: "<<reducer_sum_count<<endl;

    return 0;
}
