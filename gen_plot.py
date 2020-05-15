from glob import glob
from math import *
import matplotlib.pyplot as plt


def parse_files(file_list, ind):
	file_suf = 'Results/mpi.sub.o'
	files = [file_suf + str(i) for i in file_list] 

	results_dict = {}
	for file in files:
		with open(file, 'r') as f:
			#lines = [l.strip() for l in f.readlines() if len(l.strip()) > 0]
			lines = f.readlines()
			for i, l in enumerate(lines):

				if 'FINAL STATISTICS' in l:
					num_procs    = int(lines[i + 1].split(':')[1].strip())
					num_readers  = int(lines[i + 2].split(':')[1].strip())
					num_mappers  = int(lines[i + 3].split(':')[1].strip())
					num_reducers = int(lines[i + 4].split(':')[1].strip())
					num_writers  = int(lines[i + 5].split(':')[1].strip())
					num_files    = int(lines[i + 6].split(':')[1].strip())
					#print(file, num_procs, num_readers, num_mappers, num_reducers, num_writers, num_files,  lines[i + ind].split(':')[1].strip() + "\n")
					tot_time     = float(lines[i + ind].split(':')[1].strip())


					key_val = str(num_procs) + '|' + str(num_readers) + '|' + str(num_mappers) + '|' + str(num_reducers) + '|' + str(num_writers) + '|' + str(num_files)
					if key_val in results_dict:
						results_dict[key_val].append(tot_time)
					else:
						results_dict[key_val] = [tot_time]

	avg_dict = {}
	for key in results_dict.keys():
		avg_dict[key] = sum(results_dict[key]) / len(results_dict[key])

	return avg_dict

def get_time(res_dict, num_procs, num_readers, num_mappers, num_reducers, num_writers, num_files):
	return res_dict[str(num_procs) + '|' + str(num_readers) + '|' + str(num_mappers) + '|' + str(num_reducers) + '|' + str(num_writers) + '|' + str(num_files)]


def create_openmp(res_dict, graph_type, num_files, plot_prefix):

	thread_counts = [1, 2, 4, 8]
	if graph_type == 4:
		thread_counts = thread_counts[1:]
	one_thread_time = get_time(res_dict, 1, 1, 1, 1, 1, num_files) 

	plot_vals = []
	for t in thread_counts:
		time_val = get_time(res_dict, 1, t, t, t, t, num_files)
		if graph_type == 1:
			ylabel = 'Time Taken (In secs)'
			plot_vals.append(time_val)
		elif graph_type == 2:
			ylabel = 'Speed Up'
			speedup = one_thread_time / time_val
			plot_vals.append(speedup)
		elif graph_type == 3:
			ylabel = 'Efficiency'
			speedup = one_thread_time / time_val
			plot_vals.append(speedup / t)
		elif graph_type == 4:
			ylabel = 'Karp-Flatt Metric'
			speedup = one_thread_time / time_val
			plot_vals.append( (t*speedup - 1) / (t - 1) )

	plot_vals = [round(v, 3) for v in plot_vals]
	plt.clf()
	title = ylabel + ' vs Number of Threads' 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(thread_counts, plot_vals, '-bo')
	for xy in zip(thread_counts, plot_vals):
		ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
	plt.grid()
	plt.title(title)
	plt.xlabel('Number of Threads')
	plt.ylabel(ylabel)
	plt.savefig('Plots/' + plot_prefix + 'OpenMP_' + str(num_files) + '_'+ str(graph_type) + '_ConstPlot.png')

	print(" OpenMP " + str(ylabel))
	print(thread_counts)
	print(plot_vals)
	print("")



def create_mpi(res_dict, graph_type, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix):
	proc_nums = [1, 2, 4, 8, 16]
	if graph_type == 3 or graph_type == 4:
		proc_nums = proc_nums[1:]

	one_proc_time = get_time(res_dict, 1, 1, 1, 1, 1, num_files) 
	plot_vals = []
	for p in proc_nums:
		time_val = get_time(res_dict, p, num_readers, num_mappers, num_reducers, num_writers, num_files)
		if graph_type == 1:
			ylabel = 'Time Taken (In secs)'
			plot_vals.append(time_val)
		elif graph_type == 2:
			ylabel = 'Speed Up'
			speedup = one_proc_time / time_val
			plot_vals.append(speedup)
		elif graph_type == 3:
			ylabel = 'Efficiency'
			speedup = one_proc_time / time_val
			plot_vals.append(speedup / p)
		elif graph_type == 4:
			ylabel = 'Karp-Flatt Metric'
			speedup = one_proc_time / time_val
			plot_vals.append( (p*speedup - 1) / (p - 1) )

	plot_vals = [round(v, 3) for v in plot_vals]
	plt.clf()
	title = ylabel + ' vs Number of Processors' 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(proc_nums, plot_vals, '-bo')
	for xy in zip(proc_nums, plot_vals):
		ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
	plt.grid()
	plt.title(title)
	plt.xlabel('Number of Processors')
	plt.ylabel(ylabel)
	plt.savefig('Plots/' +  plot_prefix + 'MPI_'+str(num_readers) +  str(num_mappers) +  str(num_reducers) +  str(num_writers) + '_' + str(num_files) + '_' + str(graph_type) + '_ConstPlot.png')

	if(num_readers == 4 and num_mappers == 4 and num_reducers == 4 and num_writers == 4):
		print("MPI " + str(ylabel))
		print(proc_nums)
		print(plot_vals)
		print("")

def create_mpi_thread_times_plot(dict_tuple):
	total_dict    = dict_tuple[0]
	reader_dict   = dict_tuple[1]
	mapper_dict   = dict_tuple[2]
	sender_dict   = dict_tuple[3]
	receiver_dict = dict_tuple[4]
	reducer_dict  = dict_tuple[5]
	writer_dict   = dict_tuple[6]
	total_times    = []
	reader_times   = []
	mapper_times   = []
	sender_times   = []
	receiver_times = []
	reducer_times  = []
	writer_times   = [] 		 	
	proc_nums = [1, 2, 4, 8, 16]
	num_readers = num_mappers = num_reducers = num_writers = 4
	for p in proc_nums:
		total_times.append(get_time(total_dict, p, num_readers, num_mappers, num_reducers, num_writers, 240))
		reader_times.append(get_time(reader_dict, p, num_readers, num_mappers, num_reducers, num_writers, 240))
		mapper_times.append(get_time(mapper_dict, p, num_readers, num_mappers, num_reducers, num_writers, 240))
		sender_times.append(get_time(sender_dict, p, num_readers, num_mappers, num_reducers, num_writers, 240))
		receiver_times.append(get_time(receiver_dict, p, num_readers, num_mappers, num_reducers, num_writers, 240))
		reducer_times.append(get_time(reducer_dict, p, num_readers, num_mappers, num_reducers, num_writers, 240))
		writer_times.append(get_time(writer_dict, p, num_readers, num_mappers, num_reducers, num_writers, 240))

	map_recv_times = [x + y for x, y in zip(mapper_times, receiver_times)]
	red_wrt_times  = [x + y for x, y in zip(reducer_times, writer_times)] 

	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)

	plt.plot(proc_nums, total_times, '-bo', label = 'Total Time')
	for xy in zip(proc_nums, total_times):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')

	plt.plot(proc_nums, mapper_times, '-ro', label = 'Mapper Time')
	for xy in zip(proc_nums, mapper_times):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')

	plt.plot(proc_nums, receiver_times, '-go', label = 'Receiver Time')
	for xy in zip(proc_nums, receiver_times):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')

	plt.plot(proc_nums, map_recv_times, '-co', label = 'Mapper + Receiver Time')
	for xy in zip(proc_nums, map_recv_times):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')

	plt.plot(proc_nums,red_wrt_times, '-yo', label = 'Reducer + Writer Time')
	for xy in zip(proc_nums, red_wrt_times):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')

	plt.grid()
	title =  'Individual Thread Times vs Number of Processors for 4, 4, 4, 4 config'
	plt.title(title)
	plt.xlabel('Number of Processors')
	plt.ylabel('Time Taken (In secs)')
	plt.legend(loc='lower right')
	plt.savefig('Plots/Explanation_1.png')


def create_mpi_thread_times_plot_per_proc(dict_tuple, proc_num):
	total_dict    = dict_tuple[0]
	reader_dict   = dict_tuple[1]
	mapper_dict   = dict_tuple[2]
	sender_dict   = dict_tuple[3]
	receiver_dict = dict_tuple[4]
	reducer_dict  = dict_tuple[5]
	writer_dict   = dict_tuple[6]
	total_times    = []
	reader_times   = []
	mapper_times   = []
	sender_times   = []
	receiver_times = []
	reducer_times  = []
	writer_times   = [] 		 	
	thread_nums = [1, 2, 4, 8]
	for t in thread_nums:
		total_times.append(get_time(total_dict, proc_num, t, t, t, t, 240))
		reader_times.append(get_time(reader_dict, proc_num, t, t, t, t, 240))
		mapper_times.append(get_time(mapper_dict, proc_num, t, t, t, t, 240))
		sender_times.append(get_time(sender_dict, proc_num, t, t, t, t, 240))
		receiver_times.append(get_time(receiver_dict, proc_num, t, t, t, t, 240))
		reducer_times.append(get_time(reducer_dict, proc_num, t, t, t, t, 240))
		writer_times.append(get_time(writer_dict, proc_num, t, t, t, t, 240))

	map_recv_times = [x + y for x, y in zip(mapper_times, receiver_times)]

	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)

	plt.plot(thread_nums, total_times, '-bo', label = 'Total Time')
	for xy in zip(thread_nums, total_times):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')

	plt.plot(thread_nums, reader_times, '-yo', label = 'Reader Time')
	for xy in zip(thread_nums, reader_times):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')

	plt.plot(thread_nums, mapper_times, '-ro', label = 'Mapper Time')
	for xy in zip(thread_nums, mapper_times):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')

	plt.plot(thread_nums, reducer_times, '-go', label = 'Reducer Time')
	for xy in zip(thread_nums, reducer_times):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')

	plt.plot(thread_nums, writer_times, '-co', label = 'Writer Times')
	for xy in zip(thread_nums, writer_times):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')

	plt.grid()
	title =  'Individual Thread Times vs Number of Threads for {} processes'.format(proc_num)
	plt.title(title)
	plt.xlabel('Number of Threads')
	plt.ylabel('Time Taken (In secs)')
	plt.legend(loc='upper right')
	plt.savefig('Plots/Explanation_2_' + str(proc_num) +'.png')


def create_mpi_expl_mapper(dict_tuple):
	total_dict    = dict_tuple[0]
	reader_dict   = dict_tuple[1]
	mapper_dict   = dict_tuple[2]
	sender_dict   = dict_tuple[3]
	receiver_dict = dict_tuple[4]
	reducer_dict  = dict_tuple[5]
	writer_dict   = dict_tuple[6]

	mapper_times1 = []
	mapper_times2 = []
	mapper_times4 = []
	mapper_times8 = []

	proc_nums = [1, 2, 4, 8, 16]
	for p in proc_nums:
		mapper_times1.append(get_time(mapper_dict, p, 1, 1, 1, 1, 240))
		mapper_times2.append(get_time(mapper_dict, p, 2, 2, 2, 2, 240))
		mapper_times4.append(get_time(mapper_dict, p, 4, 4, 4, 4, 240))
		mapper_times8.append(get_time(mapper_dict, p, 8, 8, 8, 8, 240))

	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)

	plt.plot(proc_nums, mapper_times2, '-bo', label = 'For config 2, 2, 2, 2')
	for xy in zip(proc_nums, mapper_times2):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')
	plt.plot(proc_nums, mapper_times4, '-ro', label = 'For config 4, 4, 4, 4')
	for xy in zip(proc_nums, mapper_times4):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')
	plt.plot(proc_nums, mapper_times8, '-go', label = 'For config 8, 8, 8, 8')
	for xy in zip(proc_nums, mapper_times8):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')


	plt.grid()
	title =  'Mapper Times vs Number of Processors for various threads configs'
	plt.title(title)
	plt.xlabel('Number of Processors')
	plt.ylabel('Time Taken (In secs)')
	plt.legend(loc='upper right')
	plt.savefig('Plots/Explanation_3_mapper.png')

def create_mpi_expl_receiver(dict_tuple):
	total_dict    = dict_tuple[0]
	reader_dict   = dict_tuple[1]
	receiver_dict   = dict_tuple[2]
	sender_dict   = dict_tuple[3]
	receiver_dict = dict_tuple[4]
	reducer_dict  = dict_tuple[5]
	writer_dict   = dict_tuple[6]

	receiver_times1 = []
	receiver_times2 = []
	receiver_times4 = []
	receiver_times8 = []

	proc_nums = [1, 2, 4, 8, 16]
	for p in proc_nums:
		receiver_times1.append(get_time(receiver_dict, p, 1, 1, 1, 1, 240))
		receiver_times2.append(get_time(receiver_dict, p, 2, 2, 2, 2, 240))
		receiver_times4.append(get_time(receiver_dict, p, 4, 4, 4, 4, 240))
		receiver_times8.append(get_time(receiver_dict, p, 8, 8, 8, 8, 240))


	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)

	plt.plot(proc_nums, receiver_times2, '-bo', label = 'For config 2, 2, 2, 2')
	for xy in zip(proc_nums, receiver_times2):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')
	plt.plot(proc_nums, receiver_times4, '-ro', label = 'For config 4, 4, 4, 4')
	for xy in zip(proc_nums, receiver_times4):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')
	plt.plot(proc_nums, receiver_times8, '-go', label = 'For config 8, 8, 8, 8')
	for xy in zip(proc_nums, receiver_times8):
		ax.annotate('(%s, %0.2s)' % xy, xy=xy, textcoords='data')


	plt.grid()
	title =  'Receiver Times vs Number of Processors for various threads configs'
	plt.title(title)
	plt.xlabel('Number of Processors')
	plt.ylabel('Time Taken (In secs)')
	plt.legend(loc='lower right')
	plt.savefig('Plots/Explanation_3_receiver.png')



def main():

	data_set_list = [41537, 41538, 41539, 41540, 41541]

	# Plots he needed
	total_dict    = parse_files(data_set_list, 9)
	reader_dict   = parse_files(data_set_list, 11) 
	mapper_dict   = parse_files(data_set_list, 13) 
	sender_dict   = parse_files(data_set_list, 15) 
	receiver_dict = parse_files(data_set_list, 17) 
	reducer_dict  = parse_files(data_set_list, 19) 
	writer_dict   = parse_files(data_set_list, 21) 
	dict_tuple	  = (total_dict, reader_dict, mapper_dict, sender_dict, receiver_dict, reducer_dict, writer_dict)
	


	plot_prefix   = ""
	num_files = 240
	
	# OpenMP Plots
	create_openmp(total_dict, 1, num_files, plot_prefix)
	create_openmp(total_dict, 2, num_files, plot_prefix)
	create_openmp(total_dict, 3, num_files, plot_prefix)
	create_openmp(total_dict, 4, num_files, plot_prefix)	

	# MPI Plots
	num_readers  = 1
	num_mappers  = 1
	num_reducers = 1
	num_writers  = 1
	num_files    = 240 
	create_mpi(total_dict, 1, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 2, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 3, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 4, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)

	num_readers  = 2
	num_mappers  = 2
	num_reducers = 2
	num_writers  = 2
	num_files    = 240
	create_mpi(total_dict, 1, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 2, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 3, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 4, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)


	num_readers  = 4
	num_mappers  = 4
	num_reducers = 4
	num_writers  = 4
	num_files    = 240
	create_mpi(total_dict, 1, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 2, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 3, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 4, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)

	num_readers  = 8
	num_mappers  = 8
	num_reducers = 8
	num_writers  = 8
	num_files    = 240
	create_mpi(total_dict, 1, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 2, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 3, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 4, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)


	num_readers  = 8
	num_mappers  = 8
	num_reducers = 4
	num_writers  = 4
	num_files    = 240
	create_mpi(total_dict, 1, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 2, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 3, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 4, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)

	num_readers  = 4
	num_mappers  = 4
	num_reducers = 8
	num_writers  = 8
	num_files    = 240
	create_mpi(total_dict, 1, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 2, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 3, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)
	create_mpi(total_dict, 4, num_readers, num_mappers, num_reducers, num_writers, num_files, plot_prefix)	

	# Explanation Plots
	create_mpi_thread_times_plot(dict_tuple)
	create_mpi_thread_times_plot_per_proc(dict_tuple, 1)
	create_mpi_thread_times_plot_per_proc(dict_tuple, 2)
	create_mpi_thread_times_plot_per_proc(dict_tuple, 4)
	create_mpi_thread_times_plot_per_proc(dict_tuple, 8)
	create_mpi_thread_times_plot_per_proc(dict_tuple, 16)
	create_mpi_expl_mapper(dict_tuple)
	create_mpi_expl_receiver(dict_tuple)








main()