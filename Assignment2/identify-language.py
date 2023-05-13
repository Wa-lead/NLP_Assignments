import sys
import os
import time
import subprocess
import sklearn.metrics

def tokenize(path, file, save=True):
    """
    Tokenize a text file to characters and save it to a new file.
    """
    data_set_type = path.split('/')[1]
    with open(f'{path}/{file}', 'r') as f:
        text = f.read()
        text = text.lower()
        # split the text into characters
        text = list(text)
        # reconstruct the text
        text = ' '.join(text)
        if save:
            with open(f'{path}/{data_set_type}_tok/{file}.tok', 'w') as f:
                f.write(text)
        return f'{path}/{data_set_type}_tok/{file}.tok'

def train(train_dir, model_dir):
    train_base = os.listdir(train_dir)
    train_base = [f'{train_dir}/{file}' for file in train_base]
    train_base = [file for file in train_base if 'tok' not in file]

    
    for file in train_base:
        tokenize(train_dir, file)
        
        corpus, lang = train_dir.split(".")[:2]
        command = f"ngram-count -text {train_dir} -order 2 -lm {model_dir} -addsmooth 10"
        os.system(command)
    

def predict_helper(text, model_dir):
    # get the language models
    lms = os.listdir('LMs/euro_LMs')
    dir1, dir2 = text.split('/')[:2]
    path = f'{dir1}/{dir2}'
    file = text.split('/')[-1]
    text = tokenize(path,file, save=True)
    time.sleep(1)
    ppls = []
    for lm in lms:
        print('predicting for', lm, ', on', text)
        # get the language
        lang = lm.split('.')[0]
        # get the ppl of the text given the language model
        ppl = subprocess.check_output(f"ngram -lm {model_dir} -ppl {text}", shell=True)
        # get the ppl
        ppl = ppl.decode('utf-8').split('ppl=')[1].split(' ')[1]
    
        # append the ppl and language to the list
        ppls.append((ppl, lang))
    # sort the list by the pplabilities
    ppls.sort(key=lambda x: x[0])
    # return the language with the highest ppl
    return ppls[0][1]

def predict(test_dir, model_dir):
    test_files = os.listdir(test_dir)
    test_files = [f'{test_dir}/{file}' for file in test_files]
    test_files = [file for file in test_files if 'tok' not in file]
    test_files = sorted(test_files, key=lambda x: int(x.split('.')[1]))
    
    y_pred = []
    for file in test_files:
        print('predicting for', file)
        lang = predict_helper(file, model_dir)
        y_pred.append(lang)
    
    with open(test_dir+'/predictions.txt', 'w') as f:
        f.write('\n'.join(y_pred))
    return y_pred
        

def evaluate(gold, prediction):
    y_true = open(gold, 'r').read().split('\n')
    y_true = [x.split('\t')[1] for x in y_true]
    
    y_pred = open(prediction, 'r').read().split('\n')
    return sklearn.metrics.accuracy_score(y_true, y_pred)
    
    

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python identify-language.py [TRAIN|PREDICT|EVALUATE] <args>')
        sys.exit(1)

    mode = sys.argv[1]
    if mode == 'TRAIN':
        train_dir = sys.argv[2]
        model_dir = sys.argv[3]
        train(train_dir, model_dir)
    elif mode == 'PREDICT':
        test_dir = sys.argv[2]
        model_dir = sys.argv[3]
        predict(test_dir, model_dir)
    elif mode == 'EVALUATE':
        gold = sys.argv[2]
        prediction = sys.argv[3]
        evaluate(gold, prediction)
    else:
        print(f'Invalid mode: {mode}. Please choose TRAIN, PREDICT, or EVALUATE.')
        sys.exit(1)
