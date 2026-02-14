import codecs

work_dir = '/home/wangfei/study/source/dataset/'

def getIPCSim(sim1, sim2):
    sec1, sec2 = [], []
    bcls1, bcls2 = [], []
    scls1, scls2 = [], []
    sim1 = sim1.split(' ')
    for i, s in enumerate(sim1):
        if i % 3 == 0:
            sec1.append(s)
        elif i % 3 == 1:
            bcls1.append(s)
        else:
            scls1.append(s)

    sim2 = sim2.split(' ')
    for i, s in enumerate(sim2):
        if i % 3 == 0:
            sec2.append(s)
        elif i % 3 == 1:
            bcls2.append(s)
        else:
            scls2.append(s)

    def getSubCls(sim1, sim2):
        result = 0
        for s in sim1:
            if s in sim2:
                result = 1
        return result

    sec = getSubCls(sec1, sec2)
    bcls = getSubCls(bcls1, bcls2)
    scls = getSubCls(scls1, scls2)

    #return str(sec) + '\t' + str(bcls) + '\t' + str(scls)
    return str(sec) + '\t' + str(bcls) + '\t' + str(scls)

def construct_new_data(src_name, tar_name):
    sfile = work_dir + 'V6_SRC_IPC/' + src_name
    tfile = work_dir + 'V6_t_new/' + tar_name

    reader = codecs.open(filename=sfile, mode='r', encoding='utf-8')
    reader.readline()
    writer = codecs.open(filename=tfile, mode='w', encoding='utf-8')
    writer.write('Index\t#1 ID\t#2 ID\t#1 String\t#2 String\tQuality\tSection_Sim\tBClass_Sim\tSClass_Sim\n')
    inx = 0
    while True:
        line = reader.readline()
        if not line:
            break

        ss = line.strip().split('\t')

        nline = str(inx) + '\t'
        nline += ss[1] + '\t' + ss[2] + '\t'
        nline += ss[3] + ' ' + ss[5] + '\t' + ss[4] + ' ' + ss[6] + '\t'
        nline += ss[9] + '\t'
        nline += getIPCSim(ss[7], ss[8]) + '\n'
        writer.write(nline)
        writer.flush()

        if len(nline.split('\t')) != 9:
            print(123)

        inx += 1

    reader.close()
    writer.close()

def read_app_inv(fname):
    reader = codecs.open(filename=fname, mode='r', encoding='utf-8')
    result = {}
    pubId = ''
    while True:
        line = reader.readline()
        if not line:
            break

        ss = line.strip().split(':')
        if line.startswith('pubId'):
            pubId = ss[1]
            result[pubId] = set()
        elif line.startswith('data'):
            result[pubId].add(ss[1])
        else:
            continue
    reader.close()
    return result

def get_similarity(data1, data2):
    flag = 0
    for v in data1:
        if v in data2:
            flag = 1
    return flag


def construct_new_data_with_inv_app(src_name, tar_name):
    work_dir = '/home/wangfei/study/source/dataset/'
    fname = work_dir + 'V6_SRC_IPC/applicant_2010_2011_topic.txt'
    pid2apps = read_app_inv(fname)
    fname = work_dir + 'V6_SRC_IPC/inventor_2010_2011_topic.txt'
    pid2invs = read_app_inv(fname)

    ncounter = 0

    #sfile = work_dir + 'V6_SRC_IPC/' + src_name
    #tfile = work_dir + 'V6_t_inv_app/' + tar_name
    work_dir = '/home/wangfei/study/source/dataset/dataset_evaluation/src/NOTOPK_NOIPC/'
    sfile = work_dir + src_name
    tfile = work_dir + tar_name


    reader = codecs.open(filename=sfile, mode='r', encoding='utf-8')
    reader.readline()
    writer = codecs.open(filename=tfile, mode='w', encoding='utf-8')
    writer.write('Index\t#1 ID\t#2 ID\t#1 String\t#2 String\tQuality\tSection_Sim\tBClass_Sim\tSClass_Sim\tInv_Sim\tApp_Sim\n')
    inx = 0
    num = 1
    while True:
        line = reader.readline()
        if not line:
            break

        ss = line.strip().split('\t')

        nline = str(inx) + '\t'
        nline += ss[1] + '\t' + ss[2] + '\t'
        nline += ss[3] + ' ' + ss[5] + '\t' + ss[4] + ' ' + ss[6] + '\t'
        nline += ss[9] + '\t'
        nline += getIPCSim(ss[7], ss[8]) + '\t'
        qpid = ss[1]
        rpid = ss[2]
        if qpid not in pid2invs or rpid not in pid2invs:
            print('invs:' + qpid + ' ' + rpid)
            print(num)
            num += 1

        qinvs = [] if qpid not in pid2invs else pid2invs[qpid]
        rinvs = [] if rpid not in pid2invs else pid2invs[rpid]
        nline += str(get_similarity(qinvs, rinvs)) + '\t'

        if qpid not in pid2apps or rpid not in pid2apps:
            print('apps:' + qpid + ' ' + rpid)
        qapps = [] if qpid not in pid2apps else pid2apps[qpid]
        rapps = [] if rpid not in pid2apps else pid2apps[rpid]
        nline += str(get_similarity(qapps, rapps)) + '\n'
        writer.write(nline)
        writer.flush()

        ncounter += 1

        if len(nline.split('\t')) != 11:
            print(123)

        inx += 1

    print('TotalNum:' + str(ncounter))
    reader.close()
    writer.close()


if __name__ == '__main__':
    # 构建
    #infile, outfile = 'train.tsv', 'train_index.tsv'
    #infile, outfile = 'dev.tsv', 'dev_index.tsv'
    #infile, outfile = 'test.tsv', 'test_index.tsv'
    #construct_new_data(infile, outfile)

    # 构建
    #infile, outfile = 'train.tsv', 'train_index.tsv'
    infile, outfile = 'dev.tsv', 'dev_index.tsv'
    #infile, outfile = 'test.tsv', 'test_index.tsv'
    construct_new_data_with_inv_app(infile, outfile)



