#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:15:28 2022

@author: akila
"""

import os
import datetime
class Visualizer():
    def __init__(self, cache_path,checkpoint_dir, name):
        self.log_name = os.path.join(cache_path,checkpoint_dir, name, 'loss_log.txt')
        self.log_name_val = os.path.join(cache_path,checkpoint_dir, name, 'loss_log_val.txt')
        self.log_name_end_epoch = os.path.join(cache_path,checkpoint_dir, name, 'loss_log_end_epoch.txt')
        self.log_name_val_accuracy = os.path.join(cache_path,checkpoint_dir, name, 'val_accuracy.txt')        
        current_date = datetime.datetime.now()
        date_string = current_date.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_name, "a") as log_file:
            log_file.write('%s at %s\n' %('start_new_epoch',date_string))
        with open(self.log_name_end_epoch, "a") as log_file:
            log_file.write('%s at %s\n' %('start_new_epoch',date_string))
        with open(self.log_name_val, "a") as log_file:
            log_file.write('%s at %s\n' %('start_new_epoch',date_string))
        with open(self.log_name_val_accuracy, "a") as log_file:
            log_file.write('%s at %s\n' %('start_new_training',date_string))            
        
    def print_current_scalars(self, epoch, i, scalars):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in scalars.items():
            message += '%s: %.3f ' % (k, v)

        #print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    def print_current_scalars_val(self, epoch, i, scalars):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in scalars.items():
            message += '%s: %.3f ' % (k, v)

        #print(message)
        with open(self.log_name_val, "a") as log_file:
            log_file.write('%s\n' % message)
                        
    def print_scalars_endepoch(self, epoch, i, scalars):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in scalars.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name_end_epoch, "a") as log_file:
            log_file.write('%s\n' % message)    

    def print_val_accuracies(self, epoch, i, scalars):
        message = '(epoch: %d, iters: %d) \n' % (epoch, i)
        for k, v in scalars.items():
            message += '%s: %.3f ' % (k, v)

        #print(message)
        with open(self.log_name_val_accuracy, "a") as log_file:
            log_file.write('%s\n' % message)    
            
    def print_accuracy_report(self, message):
        

        #print(message)
        with open(self.log_name_val_accuracy, "a") as log_file:
            log_file.write('%s\n' % message)   