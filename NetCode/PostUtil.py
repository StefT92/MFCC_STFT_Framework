"""
POST TRAINING UTILITIES

POST PROCESSING UTILITIES.

Requires shutil

Author: S.Tomassetti <tomassetti.ste@gmail.com>, 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import time


def postPlot(network_test_output, Y_test, output_recap_file, SAVE=True, SHOW=True):
    """
    Post testing plot

    :param network_test_output:
    :param Y_test:
    :param output_recap_file: PATH of recap file: AVG and RMS Error
    :param SAVE: If True, recap file and plot are saved
    :param SHOW: If True plot is shown

    :return:
    """
    diff = np.abs(network_test_output-Y_test)
    diff_avg_plot = sum(np.abs(diff)) / diff.shape[0]
    plt.subplot(411)
    plt.stem(diff_avg_plot)
    plt.grid(True)
    plt.title('AVG Error on TEST set (parameter-wise)')
    plt.subplot(412)
    plt.stem(np.abs(np.mean(diff, axis=1)), 'black')
    plt.grid(True)
    plt.title('AVG Error on TEST set (feature set-wise)')
    plt.subplot(413)
    diff_tot = np.reshape(diff, [-1, 1])
    plt.plot(diff_tot, '+-r')
    plt.grid(True)
    plt.title('Error on TEST set (set-wise)')
    plt.subplot(414)
    plt.stem(np.reshape(network_test_output, [-1, 1]), 'b')
    plt.plot(np.reshape(Y_test, [-1, 1]), 'g+-')
    plt.grid(True)
    plt.title('Estimated output VS Test Target output')
    if SAVE:
        plt.savefig('Error on TEST set.png')
        out_file = open(output_recap_file, "w")
        out_file.write("Mean difference = %s\n" % np.mean(diff))
        out_file.write("RMS difference = %s\n" % np.sqrt(np.mean(diff ** 2)))
        out_file.close()
    if SHOW:
        plt.show()
    plt.clf()


def history_plot(history, current_directory, PLT_SHOW=True):

    hkeys = history.history.keys()
    if len(hkeys)==4:
        plt.subplot(211)
        plt.plot(history.history[hkeys[0]])
        plt.plot(history.history[hkeys[2]])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid(True)
        plt.subplot(212)
        # summarize history for loss
        plt.plot(history.history[hkeys[1]])
        plt.plot(history.history[hkeys[3]])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid(True)
        plt.savefig(current_directory+'/Loss_Fig.png')
        if PLT_SHOW:
            plt.show()
    else:
        plt.plot(history.history[hkeys[0]])
        plt.plot(history.history[hkeys[1]])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(current_directory+'/Loss_Fig.png')
        if PLT_SHOW:
            plt.show()
    plt.clf()


def history_plot_from_txt(log_file_ref, SAVE=True, SHOW=True):
    """

    :param log_file_ref: log file to print (loss and val loss)
    :param SAVE: If true fig is saved
    :param SHOW: If true fig is shown
    :return:
    """
    import re
    pattern = re.compile('loss: *')

    loss = []
    val_loss = []

    with open(log_file_ref, 'r') as f:
        log_file = f.readlines()
        for i in xrange(len(log_file) - 1):
            for match in re.finditer(pattern, log_file[i]):
                try:
                    loss.append(float(log_file[i][11:17]))
                    val_loss.append(float(log_file[i][30:36]))
                except:
                    print 'Line '+str(i)+' not shown'
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['train', 'test'], loc='upper left')
    if SAVE:
        plt.savefig('Loss_Fig.png')
    if SHOW:
        plt.show()
    plt.clf()


def systematic_test(output_folder, trained_network, output_recap_file, history_file, network_test_output, network_target_output, testing_register_type, TRAIN_TEST, NETWORK_TYPE, ERROR_AVE, PATH_VAR_INIT_ref, current_directory=os.getcwd()):
    """
    Post training or testing systematic save. Name output strut:

    processNumber_TrTe_Network_date_time_errorAve

    :param output_folder: output folder path
    :param trained_network: trained net path
    :param output_recap_file: recap avg and RMS error file
    :param history_file: loss and val_loss file ref
    :param network_test_output: .txt containing network estimation
    :param network_target_output: .txt containing network target
    :param testing_register_type: testing target set name
    :param TRAIN_TEST: 1 or 0
    :param NETWORK_TYPE: CNN or MLP
    :param ERROR_AVE: avg error
    :param current_directory:
    :param PATH_VAR_INIT_ref: .py initialization file to save in folder
    :return:
    """
    if TRAIN_TEST == 1:
        TR_TE = 'TR'
    else:
        TR_TE = 'TE'

    N_item_folder = str(len(os.listdir(output_folder)))
    if len(os.listdir(output_folder)) < 10:
        N_item_folder = '000' + N_item_folder
    elif len(os.listdir(output_folder)) >= 10 and len(os.listdir(output_folder)) < 100:
        N_item_folder = '00' + N_item_folder
    elif len(os.listdir(output_folder)) >= 100 and len(os.listdir(output_folder)) < 1000:
        N_item_folder = '0' + N_item_folder
    elif len(os.listdir(output_folder)) >= 1000:
        N_item_folder = N_item_folder

    destination_folder = output_folder + '/' + N_item_folder + '_' + TR_TE + '_' + NETWORK_TYPE + '_' + time.strftime(
        "%d-%m-%Y") + '_' + time.strftime("%H_%M_%S") + '_ERROR_AVE=' + str(ERROR_AVE)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        os.system('chmod -R 777 ' + destination_folder)

    shutil.copy2(current_directory + '/' + output_recap_file, destination_folder)
    shutil.copy2(trained_network + '.h5', destination_folder)
    if os.path.exists(trained_network + '.yaml'):
        shutil.copy2(trained_network + '.yaml', destination_folder)
    shutil.copy2(PATH_VAR_INIT_ref, destination_folder + '/PATH_VAR_INIT.py')
    if os.path.exists(current_directory + '/' + 'Error on TEST set.png'):
        shutil.copy2(current_directory + '/' + 'Error on TEST set.png', destination_folder)
    if TRAIN_TEST == 1:
        shutil.copy2(current_directory + '/' + 'Loss_Fig.png', destination_folder)
    if os.path.exists(current_directory + '/' + history_file):
        shutil.copy2(current_directory + '/' + history_file, destination_folder)
    if os.path.exists(current_directory + '/' + network_test_output):
        shutil.copy2(current_directory + '/' + network_test_output, destination_folder)
    if os.path.exists(current_directory + '/' + network_target_output):
        shutil.copy2(current_directory + '/' + network_target_output, destination_folder)
    if os.path.exists(current_directory + '/' + 'Testing_register_NAME.txt'):
        shutil.copy2(current_directory + '/' + 'Testing_register_NAME.txt', destination_folder)
    if os.path.exists(current_directory + '/' + 'MIDI_notes_Testing.txt'):
        shutil.copy2(current_directory + '/' + 'MIDI_notes_Testing.txt', destination_folder)
    if os.path.exists(current_directory + '/' + 'Networks/' + testing_register_type):
        os.makedirs(destination_folder + '/' + testing_register_type)
        os.system('chmod -R 777 ' + destination_folder + '/' + testing_register_type)
        os.system('cp -R ' + current_directory + '/' + 'Networks/' + testing_register_type + '/ ' + destination_folder + '/')

def output_file_dpckg(network_test_output, midi_note_testing_vector, testing_register_type, current_directory=os.getcwd()):
    """
    Unpack network output estimation. Write a single .txt for every note in testing, note is given by midi_note_testing_vector
    :param network_test_output: net estim
    :param midi_note_testing_vector: midi notes of network target file
    :param testing_register_type: target type (name)
    :param current_directory:
    :return:
    """
    network_test_output[:, 0] = [int(x) for x in midi_note_testing_vector]
    OUT_FOLDER_PATH = "%s%s%s" % (current_directory,'/Networks/',testing_register_type)
    if not os.path.exists(OUT_FOLDER_PATH):
        os.makedirs(OUT_FOLDER_PATH)
    PATH = "%s%s%s%s%s" % (current_directory,'/Networks/',testing_register_type,'/', testing_register_type)
    for rows in xrange(len(network_test_output)):

        if rows <10:
            tname = "%s%s%d%s" % (PATH,'_0',rows,'.txt')
        else:
            tname = "%s%s%d%s" % (PATH,'_', rows,'.txt')

        np.savetxt(tname,network_test_output[rows,:].reshape(1,len(network_test_output[rows,:])),newline=' ',fmt="%s%s" %('%3d',' %1.9e'*59))
