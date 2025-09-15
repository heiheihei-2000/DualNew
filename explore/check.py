from utils_path import candidate_path

import numpy as np
import json

def check(dataset, data_type='test'):
    """
    Check model performance on test or validation set
    Args:
        dataset: 'webqsp', 'CWQ', or 'MetaQA*'
        data_type: 'test' or 'valid' (default: 'test')
    """
    # dataset = 'webqsp'
    if dataset.startswith('MetaQA'):
        if data_type == 'valid':
            ta_file = '../data/'+ dataset + '/ntm/qa_dev.txt'
        else:
            ta_file = '../data/'+ dataset + '/ntm/qa_test.txt'
        dataset = dataset.replace('/','-')
        path_file = f'../explore/{dataset}-{data_type}-path.txt'
    elif dataset == 'webqsp':
        if data_type == 'valid':
            ta_file = '../data/webqsp/Webqsp_valid.txt'  # Assuming validation file exists
            path_file = '../explore/' + dataset + '-valid-path.txt'
        else:
            ta_file = '../data/webqsp/Webqsp.txt'
            path_file = '../explore/' + dataset + '-test-path.txt'
    elif dataset == 'CWQ':
        if data_type == 'valid':
            ta_file = '../data/CWQ/CWQ_valid.txt'  # Assuming validation file exists
            path_file = '../explore/' + dataset + '-valid-path.txt'
        else:
            ta_file = '../data/CWQ/CWQ.txt'
            path_file = '../explore/' + dataset + '-test-path.txt'

    fta = open(ta_file)  # ta: true answer
    all_candi, all_score, all_p, all_ids = candidate_path(path_file)

    all_ta = []
    n_null = 0
    for line in fta: 
        line = line.strip().split('\t')
        try:
            if dataset.startswith('WC'):
                ta = line[2]
                ta = ta.replace('/','|')[:-1]
            else:
                _, ta = line[0], line[1]
            ta = ta.strip()
        except:
            ta = 'null'
        all_ta.append(ta)

    # Use different answer file for validation set
    if data_type == 'valid':
        read_file = dataset + '-valid-ans.jsonl'
    else:
        read_file = dataset + '-ans.jsonl'
    fa = open(read_file)   
    all_a = []
    all_a_id = []
    maxid = -1
    for line in fa:
        data = json.loads(line.strip())
        id = data['id']
        id = int(id)
        if id <= maxid:
            continue
        else:
            maxid = id
            all_a.append(data['answer'])
            all_a_id.append(id)
    print(len(all_ta),len(all_a))  

    check = []
    check_abc = []
    check_A = []
    n_true = 0
    flag = 0
    for i in range(len(all_a)):
        try:
            a = all_a[i]
        except:
            a = 'null'
        i = all_a_id[i]
        ta = all_ta[i]
        if ta == 'null':
            n_null += 1
            check.append(0)
            check_abc.append(0)
            continue
        ta = ta.split('|')
        flag = 0

        for oneta in ta:
            if oneta.lower() in a.lower():
                check.append(1)
                n_true += 1
                flag = 1
                break
        if flag == 0:
            check.append(0)
        flag = 0

        # check abc 
        s = a
        index_a = s.find('A. ')
        index_b = s.find('B. ')
        index_c = s.find('C. ')
        index_d = s.find('D. ')
        index_e = s.find('E. ')
        # print(index_a, index_b, index_c)
        if i not in all_ids:
            check_abc.append(0)
            check_A.append(0)
            continue
        i = all_ids[i]
        if 0 <= index_a and (index_b ==-1 or index_b > index_a) and (index_c == -1 or index_a < index_c) and (index_d == -1 or index_a < index_d):
            a = 'A. '  + all_candi[i][0].lower()
        elif 0 <= index_b and (index_a ==-1 or index_a > index_b) and (index_c == -1 or index_b < index_c) and (index_d == -1 or index_b < index_d):
            a = 'B. '+ all_candi[i][1].lower()
        elif 0 <= index_c and (index_a ==-1 or index_a > index_c) and (index_b == -1 or index_b > index_c) and (index_d == -1 or index_c < index_d):
            a = 'C. ' + all_candi[i][2].lower()
        elif 0 <= index_d and (index_a ==-1 or index_a > index_d) and (index_b == -1 or index_b > index_d) and (index_c == -1 or index_d < index_c):
            a = 'D. ' + all_candi[i][3].lower()
        else:
            a = a.lower()

        for oneta in ta:
            if oneta.lower() in a:
                check_abc.append(1)
                # n_true += 1
                flag = 1
                break
        if flag == 0:
            check_abc.append(0)


    hit1abc = np.array(check_abc).sum() / (len(check_abc) - n_null)
    print(f'[{data_type.upper()}] HIT@1: ', hit1abc)
    hit = n_true / (len(check) - n_null)
    print(f'[{data_type.upper()}] whether the correct answer is in the reply:' , hit)  

    # Include data_type in output filename
    output_file = f'check-{data_type}-{read_file}' if data_type == 'valid' else f'check-{read_file}'
    fout = open(output_file,'w')
    for i in range(len(check)):
        data = {'id': all_a_id[i] , 'hit@1': check_abc[i], 'gold_answer': all_ta[all_a_id[i]], 'answer': all_a[i]}   
        fout.write(json.dumps(data) + '\n')
    fout.write(json.dumps({'HIT@1': hit1abc, 'HIT': hit})+ '\n')
    fout.close()


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python check.py <dataset> [data_type]")
        print("  dataset: webqsp, CWQ, or MetaQA")
        print("  data_type: test (default) or valid")
        sys.exit(1)
    
    dataset = sys.argv[1]
    data_type = sys.argv[2] if len(sys.argv) > 2 else 'test'
    
    if data_type not in ['test', 'valid']:
        print("Error: data_type must be 'test' or 'valid'")
        sys.exit(1)
    
    print(f"Evaluating {dataset} on {data_type} set...")
    check(dataset, data_type)