"""
Utils for importing modules from string
"""

import os
import datetime


def make_exp_dir(exp_dir='../experiments',prefix='',symlink_name='latest',
                 task_type=None, task_id=0):
    """
    Creates a dated directory for an experiement, and also creates a symlink 
    """
    linked_dir=exp_dir+'/%s' % symlink_name
    date_str = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    dated_dir = prefix + '%s-%s-%s' % (date_str,
                                       os.getenv('SLURM_JOB_ID'),
                                       os.getenv('SLURM_ARRAY_TASK_ID'))
    try:
        dated_dir = os.path.join(os.getenv('SLURM_ARRAY_JOB_ID'),dated_dir)
    except:
        pass
    target_dir = os.path.join(exp_dir, dated_dir)
    suffix = 1
    while os.path.exists(target_dir):
        target_dir = os.path.join(exp_dir, f"{dated_dir}-run{suffix}")
        suffix += 1
    dated_dir = os.path.basename(target_dir)
    os.makedirs(target_dir)
    if os.path.islink(linked_dir):
        os.unlink(linked_dir)
    os.symlink(dated_dir,linked_dir)
    return os.path.join(exp_dir,dated_dir)

def make_callback_dirs(logdir):
    tensorboard_dir = os.path.join(logdir, 'tboard')
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    
    checkpoints_dir = os.path.join(logdir, 'checkpoints')
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    return tensorboard_dir, checkpoints_dir
