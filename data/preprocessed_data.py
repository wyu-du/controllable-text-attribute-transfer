import os
import argparse
import nltk


def buid_dict_file(data_path):
    word_to_id = {}
    dict_file = data_path + 'processed_files/word_to_id.txt'
    file1 = ['sentiment.train.0', 'sentiment.train.1',
             'sentiment.dev.0', 'sentiment.dev.1',
             'sentiment.test.0', 'sentiment.test.1', 
             'reference.test.0', 'reference.test.1']
    for file_item in file1:
        with open(data_path + file_item, 'r') as f:
            for item in f:
                item = item.strip()
                word_list = nltk.word_tokenize(item)
                # print(word_list)
                # input("===")
                for word in word_list:
                    word = word.lower()
                    if word not in word_to_id:
                        word_to_id[word] = 0
                    word_to_id[word] += 1
    print("Get word_dict success: %d words" % len(word_to_id))
    # write word_to_id to file
    word_dict_list = sorted(word_to_id.items(), key=lambda d: d[1], reverse=True)
    with open(dict_file, 'w') as f:
        f.write("<PAD>\n")
        f.write("<UNK>\n")
        f.write("<BOS>\n")
        f.write("<EOS>\n")
        for ii in word_dict_list:
            f.write("%s\t%d\n" % (str(ii[0]), ii[1]))
            # f.write("%s\n" % str(ii[0]))
    print("build dict finished!")
    return


def build_id_file(data_path):
    # load word_dict
    word_dict = {}
    num = 0
    with open(data_path+'processed_files/word_to_id.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip()
            word = item.split('\t')[0]
            word_dict[word] = num
            num += 1
    print("Load embedding success! Num: %d" % len(word_dict))

    # generate id file
    file1 = ['sentiment.train.0', 'sentiment.train.1',
             'sentiment.dev.0', 'sentiment.dev.1',
             'sentiment.test.0', 'sentiment.test.1', 
             'reference.test.0', 'reference.test.1']
    for file_item in file1:
        id_file_data = []
        with open(data_path + file_item, 'r') as f:
            for item in f:
                item = item.strip()
                word_list = nltk.word_tokenize(item)
                # print(word_list)
                # input("===")
                id_list = []
                for word in word_list:
                    word = word.lower()
                    id = word_dict[word]
                    id_list.append(id)
                id_file_data.append(id_list)
        # write to file:
        with open(data_path+"processed_files/%s" % file_item, 'w') as f:
            for item in id_file_data:
                f.write("%s\n" % (' '.join([str(k) for k in item])))
    print('build id file finished!')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Here is your model discription.")
    parser.add_argument('--task', type=str, default='yelp', help='Specify datasets.')
    parser.add_argument('--word_to_id_file', type=str, default='', help='')
    parser.add_argument('--data_path', type=str, default='', help='')
    
    args = parser.parse_args()
    # set task type
    if args.task == 'yelp':
        args.data_path = 'yelp/processed_files/'
    elif args.task == 'amazon':
        args.data_path = 'amazon/processed_files/'
    elif args.task == 'imagecaption':
        args.data_path = 'imagecaption/processed_files/'
    else:
        raise TypeError('Wrong task type!')
    
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    buid_dict_file(args.task+'/')
    build_id_file(args.task+'/')

