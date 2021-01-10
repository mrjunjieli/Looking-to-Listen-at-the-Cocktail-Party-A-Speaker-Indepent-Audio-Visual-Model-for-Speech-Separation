base_datapath = '/Work19/2020/lijunjie/LRS3/AV_model_database/'


with open(base_datapath+'pretrain_dataset.txt', 'r') as tr:
    lines = tr.readlines()
    for line in lines:
        info = line.strip().split('.')
        num1 = info[0].strip().split('-')[1]
        num2 = info[0].strip().split('-')[2]
        new_line = line.strip() + ' ' + num1 + '_faceemb.npy' + ' ' + num2 + '_faceemb.npy\n'
        with open(base_datapath+'AVdataset_pretrain.txt', 'a') as f:
            f.write(new_line)

with open(base_datapath+'trainval_dataset.txt', 'r') as val:
    lines = val.readlines()
    for line in lines:
        info = line.strip().split('.')
        num1 = info[0].strip().split('-')[1]
        num2 = info[0].strip().split('-')[2]
        new_line = line.strip() + ' ' + num1 + '_faceemb.npy' + ' ' + num2 + '_faceemb.npy\n'
        with open(base_datapath+'AVdataset_trainval.txt','a') as f:
            f.write(new_line)

with open(base_datapath+'test_dataset.txt', 'r') as val:
    lines = val.readlines()
    for line in lines:
        info = line.strip().split('.')
        num1 = info[0].strip().split('-')[1]
        num2 = info[0].strip().split('-')[2]
        new_line = line.strip() + ' ' + num1 + '_faceemb.npy' + ' ' + num2 + '_faceemb.npy\n'
        with open(base_datapath+'AVdataset_test.txt','a') as f:
            f.write(new_line)            