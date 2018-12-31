# workspace = ""
workspace = "/home/notebooks/Cacophony/DCASE/dcase2017_task4_cvssp/"

# config
sample_rate = 8000 # from what I've seen it doesnt matter that i change this from 16000 to 8000 as long as i downsample the supplied data
n_window = 1024 #1024 / 2
n_overlap = 360 #360 / 2     # ensure 240 frames in 10 seconds
max_len = 240  #240/2       # sequence max length is 10 s, 240 frames. #f why?how to change this?I dont believe it matters for testing. just split files into 10s subfiles
#edit, 25th aug finn doesnt know if this is all algood^


step_time_in_sec = float(n_window - n_overlap) / sample_rate

# Id of classes
ids = ['/m/05zppz', '/m/03j1ly'] #, '/m/09kppz']

# Name of classes
lbs = ['Voice', 'Noise']# , 'Birdsong']
          
idx_to_id = {index: id for index, id in enumerate(ids)}
id_to_idx = {id: index for index, id in enumerate(ids)}
idx_to_lb = {index: lb for index, lb in enumerate(lbs)}
lb_to_idx = {lb: index for index, lb in enumerate(lbs)}
num_classes = len(lbs)

thres_ary = [0.1, 0.5]