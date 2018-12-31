"""
SUMMARY:  Examples of evaluation code. 
AUTHOR:   Qiuqiang Kong, q.kong@surrey.ac.uk
Created:  2017.07.10
Modified: -
--------------------------------------
"""
import os
import evaluate
import io_task4
import visualize


sample_rate = 8000 # from what I've seen it doesnt matter that i change this from 16000 to 8000 as long as i downsample the supplied data
n_window = 1024 #1024 / 2
n_overlap = 360 #360 / 2     # ensure 240 frames in 10 seconds
max_len = 240  #240/2       # sequence max length is 10 s, 240 frames. #f why?how to change this?I dont believe it matters for testing. just split files into 10s subfiles
#edit, 25th aug finn doesnt know if this is all algood^

step_sec = float(n_window - n_overlap) / sample_rate

# Name of classes
lbs = ['Voice', 'Noise']#, 'Birdsong']#todo make this more automatic
 
#strong_gt_csv = 'data/groundtruth_strong_label_testing_set_8sep.csv'#potential problem if the process expects these to have been sed'ed in the runme.sh session
#weak_gt_csv = 'data/groundtruth_weak_label_testing_set_8sep.csv'#these were changed on 16nov. not sure about ../meta_data tho

weak_gt_csv="../metadata_nov/gt_weak_nov29_testset_bgone_10000morevoice_19dec_1pc_nosing.csv"
strong_gt_csv="../metadata_nov/gt_weak_nov29_testset_bgone_10000morevoice_19dec_1pc_nosing.csv" 
#note i had a look on the 29dec and these csv var setting bits dont actually matter but not concrete on it
#weak_gt_csv="../metadata/step_files_gt.csv"
#strong_gt_csv="../metadata/step_files_gt.csv" 

#weak_gt_csv = '../meta_data/gt_weak_20.csv'#todo make this more automatic
#strong_gt_csv = '../meta_data/gt_strong_20.csv'#todo make this more automatic
#there may be problems with this groundtruth stuff


### Audio tagging evaluations. 
def at_evaluate_from_prob_mat_csv():              
    #at_prob_path = 'data/at_prob_mat.csv.gz'#read
    #at_stat_path = '_stats/at_stat.csv'#written
    #at_submission_path = '_submissions/at_submission.csv'#dec5#written
                       
    auto_thres = True
    if auto_thres:
        thres_ary = 'auto'
    else:
        thres_ary = [0.001, 0.999]#[0.5] * len(lbs)
        
    at_evaluator = evaluate.AudioTaggingEvaluate(
                       weak_gt_csv=weak_gt_csv, 
                       lbs=lbs)
        
    stat = at_evaluator.get_stats_from_prob_mat_csv(
                 pd_prob_mat_csv=at_prob_path, 
                 thres_ary=thres_ary)
                 
    at_evaluator.write_stat_to_csv(stat=stat, 
                                     stat_path=at_stat_path)
                                     
    at_evaluator.print_stat(stat_path=at_stat_path)
    
    io_task4.at_write_prob_mat_csv_to_submission_csv(at_prob_mat_path=at_prob_path, 
                                                     lbs=lbs, 
                                                     thres_ary=stat['thres_ary'], 
                                                     out_path=at_submission_path)
                                    
def at_evaluate_from_submission_csv():
    #at_submission_path = '_submissions/at_submission.csv'#read
    #at_stat_path = '_stats/at_stat_from_submission_file.csv'#dec5#written
    
    at_evaluator = evaluate.AudioTaggingEvaluate(
                       weak_gt_csv=weak_gt_csv, 
                       lbs=lbs)
    
    stat = at_evaluator.get_stats_from_submit_format(
                 submission_csv=at_submission_path)
                 
    at_evaluator.write_stat_to_csv(
        stat=stat, 
        stat_path=at_stat_path)
        
    at_evaluator.print_stat(stat_path=at_stat_path)
    
def at_evaluate_ankit():
    #at_submission_path = '_submissions/at_submission.csv'
    #ankit_csv = 'evaluation_modified_ankitshah009/groundtruth/groundtruth_weak_label_testing_set.csv'
    #at_stat_path = '_stats/at_stat_ankit.csv' #dec5
    
    at_evaluator = evaluate.AudioTaggingEvaluate(
                       weak_gt_csv=weak_gt_csv, 
                       lbs=lbs)
                       
    at_evaluator.write_out_ankit_stat(
        submission_csv=at_submission_path, 
        ankit_csv=ankit_csv, 
        stat_path=at_stat_path)
        
    at_evaluator.print_stat(at_stat_path)

### Sound event detection evaluations. 
def sed_evaluate_from_prob_mat_list_csv():
    #sed_prob_mat_list_path = 'data/sed_prob_mat_list.csv.gz'#read
    #sed_stat_path = '_stats/sed_stat.csv' #write
    #sed_submission_path = '_submissions/sed_submission.csv'#dec5 #written in sed_write_prob_mat_list_csv_to_submission_csv
    
    sed_evaluator = evaluate.SoundEventDetectionEvaluate(
                        strong_gt_csv=strong_gt_csv, 
                        lbs=lbs, 
                        step_sec=step_sec, 
                        max_len=max_len)
                        
    thres_ary = [0.001, 0.999]#[0.01] * len(lbs)
    
    stat = sed_evaluator.get_stats_from_prob_mat_list_csv(
                 pd_prob_mat_list_csv=sed_prob_mat_list_path, 
                 thres_ary=thres_ary)
                 
    sed_evaluator.write_stat_to_csv(stat=stat, 
                                      stat_path=sed_stat_path)
                                      
    sed_evaluator.print_stat(stat_path=sed_stat_path)
    
    io_task4.sed_write_prob_mat_list_csv_to_submission_csv(
        sed_prob_mat_list_path=sed_prob_mat_list_path, 
        lbs=lbs, 
        thres_ary=thres_ary, 
        step_sec=step_sec, 
        out_path=sed_submission_path)
    
def sed_evaluate_from_submission_csv():
    #sed_submission_path = '_submissions/sed_submission.csv'#read
    #sed_stat_path = '_stats/sed_stat_from_submission.csv'#dec5#written
    
    sed_evaluator = evaluate.SoundEventDetectionEvaluate(
                        strong_gt_csv=strong_gt_csv, 
                        lbs=lbs, 
                        step_sec=step_sec, 
                        max_len=max_len)
                        
    stat = sed_evaluator.get_stats_from_submit_format(
                 submission_csv=sed_submission_path)
                 
    sed_evaluator.write_stat_to_csv(stat=stat, 
                                      stat_path=sed_stat_path)
                                      
    sed_evaluator.print_stat(stat_path=sed_stat_path)
    
### Visualizations. 
def at_visualize():
    #at_prob_mat_path = 'data/at_prob_mat.csv.gz'#read
    #out_path = '_visualizations/at_visualization.csv'#dec5#written
    
    visualize.at_visualize(at_prob_mat_path=at_prob_mat_path, 
                           weak_gt_csv=weak_gt_csv, 
                           lbs=lbs, 
                           out_path=out_path)

def sed_visualize():
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    #sed_prob_mat_list_path = 'data/sed_prob_mat_list.csv.gz'#dec5#read
    
    (na_list, pd_prob_mat_list, gt_digit_mat_list) = visualize.sed_visualize(
         sed_prob_mat_list_path=sed_prob_mat_list_path, 
         strong_gt_csv=strong_gt_csv, 
         lbs=lbs, 
         step_sec=step_sec, 
         max_len=max_len)
         
    for n in range(len(na_list)):
        na = na_list[n]
        pd_prob_mat = pd_prob_mat_list[n]
        gt_digit_mat = gt_digit_mat_list[n]
        
        fig, axs = plt.subplots(3, 1, sharex=True)
        
        axs[0].set_title(na + "\nYou may plot spectrogram here yourself. ")
        # axs[0].matshow(x.T, origin='lower', aspect='auto') # load & plot spectrogram here. 
        
        axs[1].set_title("Prediction")
        axs[1].matshow(pd_prob_mat.T, origin='lower', aspect='auto', vmin=0., vmax=1.)	
        axs[1].set_yticklabels([''] + lbs)
        axs[1].yaxis.set_major_locator(ticker.MultipleLocator(1))
        axs[1].yaxis.grid(color='w', linestyle='solid', linewidth=0.3)
        
        axs[2].set_title("Ground truth")
        axs[2].matshow(gt_digit_mat.T, origin='lower', aspect='auto', vmin=0., vmax=1.)	
        axs[2].set_yticklabels([''] + lbs)
        axs[2].yaxis.set_major_locator(ticker.MultipleLocator(1))
        axs[2].yaxis.grid(color='w', linestyle='solid', linewidth=0.3)
        fig.savefig(sed_vis)
        
    
### main
if __name__ == '__main__':
    #note i had a look on the 29dec and these csv var setting bits dont actually matter... but not concrete on it
    weak_gt_csv="../metadata_nov/gt_weak_nov29_testset_bgone_10000morevoice_19dec_1pc_nosing.csv"
    strong_gt_csv="../metadata_nov/gt_weak_nov29_testset_bgone_10000morevoice_19dec_1pc_nosing.csv"  
    #weak_gt_csv = '../meta_data/gt_weak_20.csv'#todo make this more automatic
    #strong_gt_csv = '../meta_data/gt_strong_20.csv'#todo make this more automatic
    #weak_gt_csv="../metadata/step_files_gt.csv"
    #strong_gt_csv="../metadata/step_files_gt.csv" 
    
    
    
    if not os.path.exists('_submissions'): os.makedirs('_submissions')
    if not os.path.exists('_stats'): os.makedirs('_stats')
    
    at_type = 1
    sed_type = 1
    ankit_evaluate = False
    at_vis = True
    sed_vis = True
    
        
    print("============= Audio tagging Evaluation =============")
    if at_type == 0:
        at_prob_mat_path = '../preds/crnn_sed/at_prob_mat.csv.gz'
        at_submission_path = '_submissions/at_submission.csv'
        at_stat_path = '_stats/at_stat.csv'
        at_evaluate_from_prob_mat_csv()
    elif at_type == 1:
        at_submission_path = '../submissions/crnn_sed/at_submission.csv'
        at_stat_path = '_stats/at_stat.csv'
        at_evaluate_from_submission_csv()
        
    if ankit_evaluate:
        at_evaluate_ankit()
    
    print("============= Sound event detection Evaluation =============")
    if sed_type == 0:
        
        sed_prob_mat_list_path = '../preds/crnn_sed/sed_prob_mat_list.csv.gz'
        sed_submission_path = '_submissions/sed_submission.csv'#write
        sed_stat_path = '_stats/sed_stat.csv'#write
        sed_evaluate_from_prob_mat_list_csv()
    elif sed_type == 1:
        sed_submission_path = '../submissions/crnn_sed/sed_submission.csv'#read
        sed_stat_path = '_stats/sed_stat.csv'#write
        sed_evaluate_from_submission_csv()

    print("============= Visualizations =============")
    if at_vis:
        at_prob_mat_path = '../preds/crnn_sed/at_prob_mat.csv.gz'
        out_path = '_visualizations/at_visualization.csv'
        at_visualize()
        
    if sed_vis:
        sed_prob_mat_list_path = '../preds/crnn_sed/sed_prob_mat_list.csv.gz'
        sed_vis = '_visualizations/sed_vis.png'#have to chaneg this if you want copies!
        sed_visualize()
