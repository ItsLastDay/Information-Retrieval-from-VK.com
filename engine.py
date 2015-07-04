# -*- coding: utf-8 -*-
import json
from codecs import open as copen
from nltk import word_tokenize
from subprocess import check_output
from re import sub
from sys import exc_info
from itertools import groupby
import time
import random
from math import log, sqrt

__author__ = 'Mikhail Koltsov'


class MyStemWrapper:
    def __init__(self):
        self._mystem = 'engine/mystem'
        self._has_punct = lambda x: not x.isalpha()

    def _remove_questions(self, lst):
        ret = []

        for word in lst:
            if word and word[-1] == '?':
                ret.append(word[:-1])
            else:
                ret.append(word)

        return ret

    def bulk_lemmatize(self, tokens):
        '''
            Returns list of pairs (token, lemmas). Just one call to mystem is used.
        '''
        tokens = list(filter(lambda x: x != '' and not self._has_punct(x), tokens))
        output = check_output([self._mystem, "-nl"], universal_newlines=True,\
                input='\n'.join(tokens))

        ret = []
        for (line, word) in zip(output.split('\n'), tokens):
            line = line.strip().split('|')
            line = self._remove_questions(line)
            ret.append((word, line))

        return ret

    def lemmatize(self, token):
        '''
            Returns list of possible lemmas of token using MyStem library.
        '''
        if self._has_punct(token):
            # mystem thinks there is >1 words if '-' or ' ' are presented
            return [token]
        output = check_output([self._mystem, "-nl"], universal_newlines=True, input=token)

        output = output.strip().split('|')
        output = self._remove_questions(output)

        return output


class InverseIndex:
    def __init__(self, document_list):
        self._index = dict()
        self._synonims = dict()
        self._lemmatized = dict()
        self._lemmatizer = MyStemWrapper()
        self._raw = document_list
        texts = ['%s %s %s' % (doc['t'], doc.get('lte', ''), doc.get('lti', '')) for doc in document_list]
        self._number_of_documents = len(document_list)
        self._norms = dict()
        self._term_freq = [dict() for i in range(self._number_of_documents)]
        print('Started index building...')
        self._build(texts)

    def _add_occurence(self, term, docNum):
        if term != '':
            if term not in self._index:
                self._index[term] = []
            if len(self._index[term]) == 0 or self._index[term][-1] != docNum:
                self._index[term].append(docNum)
            self._term_freq[docNum][term] = self._term_freq[docNum].get(term, 0) + 1

    def get_idf(self, term):
        return log((self._number_of_documents * 1.0) / (1 + len(self.get_posting(term))))

    def get_tf(self, term, doc_id):
        return self._term_freq[doc_id].get(term, 0)

    def get_tf_idf(self, term, doc_id):
        return self.get_tf(term, doc_id) * self.get_idf(term)

    def calc_norm(self, doc_id):
        if doc_id in self._norms:
            return self._norms[doc_id]
        else:
            ret = 0

            for word in self._term_freq[doc_id]:
                ret += self.get_tf_idf(word, doc_id) ** 2

            ret = sqrt(ret * 1.0)
            self._norms[doc_id] = ret

            return ret

    def _clean(token):
        '''
            Instead of stemming, tokens are cleaned: trailing and beginning 
            punctuation is removed, all letters are turned to lowercase.
        '''
        if token.istitle():
            return token

        token = token.lower()
        j = len(token) - 1
        i = 0
        while j >= 0 and not token[j].isalnum():
            j -= 1
        while i <= j and not token[i].isalnum():
            i += 1
        if i > j:
            return ''
        return token[i:j + 1]

    def _lemmatize(self, token):
        if token not in self._lemmatized:
            self._lemmatized[token] = self._lemmatizer.lemmatize(token)
        return self._lemmatized[token]

    def _bulk_lemmatize(self, tokens):
        result = self._lemmatizer.bulk_lemmatize(tokens)
        for (token, lemmas) in result:
            self._lemmatized[token] = lemmas

    def _build(self, doc_list):
        '''
            Constructs inverse index (based on documents in @doc_list)
            in a form of map <string> -> <int, list>:
            <string> is a term, which is presented in some document in @doc_list;
            <list> is a list of documents (their numbers) in which <string> is 
            presented;
            <int> is a length of <list>.
        '''
        all_tokens = set()
        for i in range(len(doc_list)):
            tokens = word_tokenize(doc_list[i])

            tokens = [InverseIndex._clean(t) for t in tokens]
            all_tokens.update(set(tokens))

            doc_list[i] = tokens

        self._bulk_lemmatize(list(all_tokens))

        for i in range(len(doc_list)):
            for term in doc_list[i]:
                for tt in self._lemmatize(term):
                    self._add_occurence(tt, i)
                self._add_occurence(term, i)
        
        for term, posting_list in self._index.items():
            self._index[term] = (len(posting_list), posting_list)

    def get_posting(self, term):
        return self._index.get(term.lower(), (0, []))

    def get_index(self):
        return self._index

    def register_synonims(self, synonims):
        print('Started registering synonims in index...')
        all_tokens = set()
        for (word, syn_list) in synonims:
            all_tokens.update(set(word))
            all_tokens.update(set(filter(lambda x: ' ' not in x, syn_list)))
        self._bulk_lemmatize(list(all_tokens))

        ander = lambda x: '(%s)' % ' AND '.join(x)
        orer = lambda x: '(%s)' % ' OR '.join(x)
        for (word, syn_list) in synonims:
            self._synonims[word] = []

            for syn in syn_list:
                syn = syn.strip()
                syn = sub('\s+', ' ', syn)
                if ' ' in syn:
                    self._synonims[word].append(ander(syn.split(' ')))
                else:
                    self._synonims[word].append(orer(self._lemmatize(syn)))
            self._synonims[word] = orer(self._synonims[word])
        print('Finished registering synonims in index...')


class QueryProcessor:
    def __init__(self):
        self._index = None

    def set_index(self, index):
        self._index = index

    def _query_not(self, posting):
        ''' Performs NOT operation on posting, which is a pair
            (length_of_list, posting_list).
            Returns posting list in same form.
        '''
        result = []
        lst = [-1] + posting[1] + [self._index._number_of_documents]

        for i in range(1, posting[0] + 2):
            result.extend(range(lst[i - 1] + 1, lst[i]))

        return (len(result), result)

    def _query_and(self, x, y):
        ''' Performs AND operation on postings x and y, which are pairs
            (length_of_list, posting_list).
            Returns posting list in same form.
        '''
        len_x = x[0]
        j, len_y = 0, y[0]
        result = []

        for i in range(len_x):
            while j < len_y and y[1][j] < x[1][i]:
                j += 1
            if j == len_y:
                break

            if y[1][j] == x[1][i]:
                result.append(x[1][i])

        return (len(result), result)


    def _query_or(self, x, y):
        ''' Performs OR operation on postings x and y, which are pairs
            (length_of_list, posting_list).
            Returns posting list in same form.
        '''
        len_x = x[0]
        j, len_y = 0, y[0]
        result = []

        for i in range(len_x):
            while j < len_y and y[1][j] < x[1][i]:
                result.append(y[1][j])
                j += 1
            if j == len_y:
                result.extend(x[1][i:])
                break

            result.append(x[1][i])

            if y[1][j] == x[1][i]:
                j += 1

        result.extend(y[1][j:])

        return (len(result), result)

    def _get_chain(self, query, follow_syn):
        ''' Parses query and returns (cmd, postings), where
            cmd = OR or AND
            postings = list of pairs (length_of_list, posting_list)
        '''
        cmd = ''
        plist = []

        start = 0
        length = len(query)

        while start < length:
            while start < length and query[start] == ' ':
                start += 1

            # parse expression [begin, end)
            begin, end = start, start
            if query[start] == '(':
                # (E)
                begin = start + 1
                balance = 1
                start += 1

                # need to find matching ')' bracket
                while start < length:
                    if query[start] == ')':
                        balance -= 1
                    elif query[start] == '(':
                        balance += 1

                    if balance == 0:
                        end = start
                        break
                    start += 1

                if balance != 0:
                    raise ValueError('Unmatched bracket in query ' + query)
            else:
                # t
                while start < length and query[start] != ' ':
                    start += 1
                end = start

            plist.append(self._execute_query(query[begin:end], follow_syn))
            start = min(end, length - 1) + 1

            # now read command (if presented)
            while start < length and query[start] == ' ':
                start += 1

            if start != length:
                cmd_buf = query[start:start + 4]
                is_valid_cmd = True

                if cmd_buf.startswith('AND'):
                    if cmd == 'AND' or cmd == '':
                        cmd = 'AND'
                        start += 3
                    else:
                        is_valid_cmd = False
                elif cmd_buf.startswith('OR'):
                    if cmd == 'OR' or cmd == '':
                        cmd = 'OR'
                        start += 2
                    else:
                        is_valid_cmd = False
                else:
                    print(cmd_buf)
                    raise ValueError('AND\OR expected in query ' + query)

                if not is_valid_cmd:
                    raise ValueError('You should put either only AND or ' +\
                            'only OR between expressions in query ' + query)

        if len(plist) < 2 and cmd != '':
            raise ValueError('Not enough arguments for ' + cmd + ' in query '\
                    + query)

        return (cmd, plist)

    def _execute_query(self, query, follow_syn):
        '''
            Returns (length_of_list, posting_list) that matches query.
            The grammar describing valid queries is:
                E -> t | NOT (E) | S AND S | G OR G
                S -> t | (E) | S AND S
                G -> t | (E) | G OR G
            (t stands for any term)
            where terminals are {t, NOT, (, ), AND, OR},
            non-terminals are {E, S, G}
            and the initial state is E.

            Valid query examples:
                me OR (kitchen) OR (silk AND (NOT (carpet)))
                NOT (NOT (NOT (баран)))
                (fire AND stone) OR (cover AND BrIbEr)
                NOTORious
                hi AND (NOT (bye))
            Invalid query examples:
                (tower)
                NOT ())
                well))

                a b
                carpet AND colourful OR dog AND fluffy
                carpet AND NOT (brilliant)

            Implementation does not follow grammar rules, instead it does:
                E -> t | NOT (E) | CHAIN
            where CHAIN means (e_1 AND\OR e_2 AND\OR ... AND\OR e_k).
            The operator between expressions must be the same.
            E_i here can stand for (E) or t.
            CHAIN expressions are calculated separately and are reordered
            by length of posting lists (ascendingly for AND, descendingly for OR),
            then they are proceeded left-to-right.
        '''
        query = query.strip()
        if query == '':
            raise ValueError('Empty subquery!')

        if ' ' not in query and not query.startswith('('):
            # simple token
            ret = (0, [])
            query = InverseIndex._clean(query)
            for query in (self._index._lemmatize(query) + [query]):
                ret = self._query_or(ret, self._index.get_posting(query))

                if follow_syn and query in self._index._synonims:
                    # print('SUBSTITUTING', query, 'FOR', self._index._synonims[query.lower()])
                    ret = self._query_or(ret, self._execute_query(\
                        self._index._synonims[query], False))

            return ret

        elif query.startswith('NOT'):
            # NOT (E)
            query = query[3:].strip()
            if not query.startswith('(') or not query.endswith(')'):
                raise ValueError('A part of query does not match rules: NOT ' + query)
            return self._query_not(self._execute_query(query[1:-1], follow_syn))
        else:
            # CHAIN
            cmd, postings = self._get_chain(query, follow_syn)
            postings.sort(reverse=cmd == 'OR')

            result = postings[0]
            for i in range(1, len(postings)):
                if cmd == 'OR':
                    result = self._query_or(result, postings[i])
                else:
                    result = self._query_and(result, postings[i])
            return result

    def score(self, doc_id, word_list):
        q_tf = dict()

        for words in word_list:
            cost = 2
            if words.endswith('_#!'):
                words = words[:-3]
                cost = 10
            for word in self._index._lemmatize(words):
                q_tf[word] = q_tf.get(word, 0) + cost / 2
            q_tf[words] = q_tf.get(words, 0) + cost

        #print('Initial words list: ', word_list)
        #print('Scoring for words: ', q_tf)
        #print('Doc_id = %d' % doc_id)
        #print('Doc text is %s' % self._index._raw[doc_id]['t'])
        q_norm = 0
        doc_norm = self._index.calc_norm(doc_id)
        dot_prod = 0

        for word in q_tf:
            q_tf[word] *= self._index.get_idf(word)
            q_norm += q_tf[word] ** 2

            dot_prod += q_tf[word] * self._index.get_tf_idf(word, doc_id)

        q_norm = sqrt(q_norm)

        #print('Resulting norms are: dp = %f, qn = %f, dn = %f, score = %f' % (dot_prod, q_norm, doc_norm, dot_prod / (q_norm + 1e-9) / (doc_norm + 1e-9)))
        #print('')

        vector_model_rank = dot_prod / (q_norm + 1e-9) / (doc_norm + 1e-9)

        time_rank = self._index._raw[doc_id]['d'] / 1419262169.419906 # 22 Dec 2014 ~ current time

        normalized_attractivity = (self._index._raw[doc_id]['l'] + \
                self._index._raw[doc_id]['r'] * 2 + \
                self._index._raw[doc_id]['c'] * 3.0) / self._index._raw[doc_id]['gc']

        return (vector_model_rank + time_rank + normalized_attractivity) / 3

    def similarity(self, doc1, doc2):
        if len(self._index._term_freq[doc1]) > len(self._index._term_freq[doc2]):
            doc1, doc2 = doc2, doc1

        dp = 0
        for word in self._index._term_freq[doc1]:
            dp += self._index.get_tf(word, doc1) * self._index.get_tf(word, doc2)

        cosine = 1 - dp / self._index.calc_norm(doc1) / self._index.calc_norm(doc2)

        return cosine

    def query(self, query):
        """
            Returns sorted list of pairs (docID, snippet), sorted by docID.
        """
        query_repr = query.lower()

        query = query.replace(' ', ' AND ')
        postings = self._execute_query(query, True)[1]
        result = []

        query_word_list = []
        for q in query_repr.strip().split(' '):
            q = q.strip()
            if q == '':
                continue

            syn = self._index._synonims.get(q, '')
            syn = syn.replace('(', ' ').replace(')', ' ').replace(' OR ', ' ').strip()
            syn = sub('\w+ AND \w+', ' ', syn)
            syn = syn.strip().split(' ')

            for word in syn:
                word = word.strip()
                if word != '':
                    query_word_list.append(word.lower().strip())
            query_word_list.append(q.lower() + '_#!')

        order = [(self.score(doc, query_word_list), doc) for doc in postings]

        order.sort(reverse=True)

        sz = len(order)
        marked = [False for i in range(sz)]
        
        for i in range(min(sz, 100)):
            if marked[i]:
                continue
            if order[i][0] < 0.001:
                break
            cl = [self._index._raw[order[i][1]]]

            for j in range(i + 1, min(sz, 100)):
                if not marked[j] and  self.similarity(order[i][1], order[j][1]) > 0.8:
                    cl.append(self._index._raw[order[j][1]])
                    marked[j] = True

            result.append(cl)

        return result


def get_documents():
    print('Started reading database file...')
    BOOK_NAME = 'engine/parsed_vkapi.txt'
    data_reader = open(BOOK_NAME, 'r')
    raw_data = data_reader.read().strip()
    data_reader.close()

    data = json.loads(raw_data)

    max_volume = 10 ** 5

    grouped_gid = dict()

    for d in data:
        idx = d['gid']
        if idx not in grouped_gid:
            grouped_gid[idx] = []
        grouped_gid[idx].append(d)

    data = []
    left = dict()
    right = dict()

    for gid in grouped_gid:
        grouped_gid[gid].sort(key = lambda d: d['d'], reverse=True)
        left[gid] = 0
        right[gid] = len(grouped_gid[gid])

    while max_volume > 0:
        any_has = False
        for gid in grouped_gid:
            if left[gid] < right[gid]:
                any_has = True
                max_volume -= 1
                data.append(grouped_gid[gid][left[gid]])
                left[gid] += 1
        if not any_has:
            break
            
    grouped_gid = None
    left = right = None

    print('Finished reading database file!') 
    
    return data


def get_synonims():
    """
        Retuns list of pair (word, list_of_synonims).
    """
    print('Started reading synonims file...')
    SYN_NAME = 'engine/synonims.txt'

    data_reader = copen(SYN_NAME, 'r', 'windows-1251')
    raw_data = data_reader.read()
    data_reader.close()

    data = raw_data.split('\r\n')
    data = list(map(lambda x: tuple(x.split('|')), data))
    # interested in synonims to one word precisely:
    data = list(filter(lambda x: '?' not in x[0] and ' ' not in x[0], data)) 

    ret = []
    for line in data:
        if len(line) < 2:
            continue
        word = InverseIndex._clean(line[0])
        syno = line[1]
        syno = syno.split(',')
        # don't bother with multiple choices, like 'you (shall, will)'
        syno = list(filter(lambda x: ')' not in x and '(' not in x, syno))
        syno = list(map(lambda x: InverseIndex._clean(x), syno))
        
        ret.append((word, syno))
        
    print('Finished reading synonims file!')
    return ret


def initialize():
    start = time.time()
    print('Starting index initialisation...')
    indexer = InverseIndex(get_documents())
    print('Finished building index!')
    indexer.register_synonims(get_synonims())
    q = QueryProcessor()
    q.set_index(indexer)
    print('Finished indexing initialization in %f minutes' % ((time.time() - start) / 60))
    return q # example usage: q.query('курс доллара и евро')
