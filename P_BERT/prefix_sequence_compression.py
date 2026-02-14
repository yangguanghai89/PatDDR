import torch, random

class Pipeline():
    """ Preprocess Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError

class Tokenizing_with_position_count_for_prefixSPE(Pipeline):
    ''' Tokening sentence pair '''
    def __init__(self, tokenizer):
        super().__init__()
        self.preprocessor = tokenizer.convert_to_unicode  # e.g. text normalization
        self.tokenize = tokenizer.tokenize  # tokenize function
        self.basic_tokenize = tokenizer.basic_tokenizer.tokenize

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)
        token_a, soft_position_a, token_count_a = self.tokenize_with_position_count(text_a)
        token_b, soft_position_b, token_count_b = self.tokenize_with_position_count(text_b)

        return (label, token_a, soft_position_a, token_count_a, token_b, soft_position_b, token_count_b)


    def tokenize_with_position_count(self, full_sequnece):
        term2count = {}
        for term in self.basic_tokenize(full_sequnece):
            if term not in term2count.keys():  #
                term2count[term] = 0
            term2count[term] += 1

        split_tokens = []
        soft_position = []
        token_count = []

        term2einx = {}
        preterm = None

        for term in self.basic_tokenize(full_sequnece):

            if term in term2einx.keys():  # 不记录重复项
                preterm = term
                continue

            if preterm == None:
                preterm_inx = 0
            else:
                preterm_inx = term2einx[preterm]
            cur_inx = preterm_inx
            for token in self.tokenize(term):
                split_tokens.append(token)
                token_count.append(term2count[term])
                cur_inx += 1
                soft_position.append(cur_inx)

            term2einx[term] = cur_inx  # 信息归档
            preterm = term

        return split_tokens, soft_position, token_count


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, soft_position_a, token_count_a, tokens_b, soft_position_b, token_count_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2

        truncate_tokens_pair(tokens_a, tokens_b, _max_len)
        truncate_tokens_pair(soft_position_a, soft_position_b, _max_len)
        truncate_tokens_pair(token_count_a, token_count_b, _max_len)

        tokens = []
        segment_ids = []
        soft_position = []
        token_count = []

        max_position_a = 0

        tokens.append("[CLS]")
        segment_ids.append(0)
        soft_position.append(0)
        token_count.append(1)
        for i, token in enumerate(tokens_a):
            tokens.append(token)
            segment_ids.append(0)
            soft_position.append(soft_position_a[i])
            max_position_a = soft_position_a[i] if soft_position_a[i] > max_position_a else max_position_a
            token_count.append(token_count_a[i])
        tokens.append("[SEP]")
        segment_ids.append(0)
        max_position_a += 1
        soft_position.append(max_position_a)
        token_count.append(2)

        max_position_b = max_position_a
        if tokens_b:
            for i, token in enumerate(tokens_b):
                tokens.append(token)
                segment_ids.append(1)
                soft_position.append(soft_position_b[i] + max_position_a)  # 确保soft_position_b的位置与 soft_position_a不重合
                max_position_b = (soft_position_b[i] + max_position_a) if (soft_position_b[i] + max_position_a) > max_position_b else max_position_b
                token_count.append(token_count_b[i])
            tokens.append("[SEP]")
            segment_ids.append(1)
            max_position_b += 1
            soft_position.append(max_position_b)
            token_count.append(2)

            # 如果某一单词的数目高于128,则直接设置为127
            for i, tc in enumerate(token_count):
                if tc >= self.max_len:
                    # print(tokens_a)
                    # print(tokens_b)
                    # print("Error: 字符{}数目为{}！".format(tokens[i], tc))
                    token_count[i] = self.max_len - 1
                    # exit(0)

        return (label, tokens, segment_ids, soft_position, token_count)

class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens, segment_ids, soft_position, token_count = instance

        input_ids = self.indexer(tokens)
        input_mask = [1]*(len(tokens))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        soft_position.extend([0] * n_pad)
        token_count.extend([0] * n_pad)
        input_mask.extend([0]*n_pad)

        return (input_ids, segment_ids, soft_position, token_count, input_mask, label_id)

def handle_trainDataset(text_a, text_b, pipelines):
    data = []
    for ta, tb in zip(text_a, text_b):
        # 数据增强
        instances = dataAugmentation(('0', ta, tb), ratio=2)
        for instance in instances:
            for proc in pipelines:  # a bunch of pre-processing
                instance = proc(instance)
            data.append(instance)

    # To Tensors
    return [torch.tensor(x, dtype=torch.long) for x in zip(*data)]


def handle_evalDataset(text_a, text_b, pipelines):
    data = []
    for ta, tb in zip(text_a, text_b):
        instance = ('0', ta, tb)
        for proc in pipelines:
            instance = proc(instance)
        data.append(instance)

    # To Tensors
    return [torch.tensor(x, dtype=torch.long) for x in zip(*data)]


def handle_testDataset(labels, text_a, text_b):
    global pipeline
    data = []
    for label, ta, tb in zip(labels, text_a, text_b):
        instance = (label, ta, tb)
        for proc in pipeline:
            instance = proc(instance)
        data.append(instance)

    # To Tensors
    return [torch.tensor(x, dtype=torch.long) for x in zip(*data)]


def dataAugmentation(instance, ratio=2):
    instances = list()
    label, text_a, text_b = instance
    text_a = text_a.split(' ')
    text_b = text_b.split(' ')
    # val = min(len(text_a), len(text_b))
    for i in range(ratio):
        if i == 0:
            instances.append(instance)
        else:
            val = len(text_a)
            bndy = random.randint(2, val - 1)
            text_a = text_a[bndy:len(text_a)] + text_a[0:bndy]
            val = len(text_b)
            bndy = random.randint(2, val - 1)
            text_b = text_b[bndy:len(text_b)] + text_b[0:bndy]
            instances.append((label, ' '.join(text_a), ' '.join(text_b)))
    return instances