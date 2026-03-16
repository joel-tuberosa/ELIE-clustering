'''
This module contains classes and functions to handle and integrate 
information extracted from specimen labels. It allows to build 
searchable text databases using token extraction and text feature 
scoring. This module uses the packages regex, sklearn, nltk and leven.
'''

import json, elieclustering.date, regex, sys
from nltk import regexp_tokenize
from elieclustering.utils import (mismatch_rule, 
                        get_word_tokenize_pattern, 
                        strip_accents, 
                        get_norm_leven_dist)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from leven import levenshtein
from collections import defaultdict

# =============================================================================
# CLASSES
# -----------------------------------------------------------------------------
class Label(object):
    '''
    Store raw label data along with an identifier.
    '''
    
    keys = ("ID", "text")
    
    def __init__(self, ID=None, text=None, **kwargs):
        '''
        Instantiate a new Label object from a unique identifier and a 
        transcript.

        Parameters
        ----------
            ID : str
                A unique identifier.
            
            text : str
                Any text, ideally a label transcript.
            
            **kwargs
                Extra keyword arguments are ignored. This allows Label 
                objects to be created from JSON data with additional 
                fields (e.g., confidence scores).
        '''
        
        self._data = {"ID": ID, "text": text}
    
    def export(self):
        '''
        Returns a dict object with the attribute data.
        '''
        
        return dict(**self._data)
    
    def get_tuple(self, keys=None):
        '''
        Returns a tuple with the attributes values.

        Parameters
        ----------
            key : list
                The list of attributes to be put in the tuple.
        '''
        
        if keys is None: keys=self.keys
        return tuple( self._data[key] for key in keys ) 
    
    @property
    def ID(self):
        return self._data["ID"]
    
    @property
    def text(self):
        return self._data["text"]
    
    def __hash__(self):
        return hash(self.get_tuple())

    def __repr__(self):
        return f'Label(ID: {self.ID}, text: {repr(self.text)})'
    
class CollectingEvent(Label):
    '''
    Store collecting event data, which include a unique identifier, a 
    location, a date, a collecting entity and representative label
    transcript.
    '''
    
    ### implement a function to identity the different elements of a label text
    ### and make a collecting event object from a label object
    
    keys = ("ID", "location", "date", "collector", "text")
    
    def __init__(self, ID=None, location=None, date=None, collector=None, 
                 text=None):
        '''
        Instanciate a CollectingEvent object with the provided 
        attribute values.

        Parameters
        ----------
            ID : str
                A unique identifier.

            location : str
                Any text, ideally a standardized address referring to 
                the collecting event location. 
            
            date : str
                Any text, ideally a date or a date range referring to 
                the collecting event time.
            
            collector : str
                Any text, ideally the name of the person (or people) or
                the entity responsible for this collecting event.
            
            text : str
                Any text, ideally the reference label transcript for 
                this collecting event.

        '''
        
        self._data = {"ID": ID, 
                      "location": location, 
                      "date": date,
                      "collector": collector,
                      "text": text}
    
    @property
    def location(self):
        return self._data["location"]
    
    @property
    def date(self):
        return self._data["date"]
    
    @property
    def collector(self):
        return self._data["collector"]
    
    def __repr__(self):
        return (f'CollectingEvent(ID: {self.ID},'
                f' Location: {self.location},'
                f' Date: {self.date}, Collector: {self.collector},'
                f' text: {repr(self.text)}')

class Mask(object):
    '''
    Store a serie of regular expressions attached to attribute names. 
    Mask objects are used to remove matching character strings in the
    corresponding attributes of a given object (i.e. Label or 
    CollectingEvent), prior to build a database.
    '''
    
    def __init__(self, **kwargs):
        '''
        Build the Mask object from linked attribute names and regular 
        expressions.

        Parameters
        ----------
            * : str
                A regular expression pattern to be compiled and linked
                to the attribute of the same name as the provided 
                keyword argument. 
        '''

        # compile regular expression and build property
        self._data = dict()
        for key in kwargs:
            expr = kwargs[key]
            if type(expr) is str:
                expr = regex.compile(expr)
            self._data[key] = expr
    
    def get_masked_str(self, target, attr):
        '''
        Get the text corresponding to the provided attribute in the 
        provided target object. Remove the characters matching the 
        corresponding regular expression and returns the cleaned text. 
        
        Parameters
        ----------
            target : Label
                A Label object or alike.

            attr : str
                The name of an attribute of the provided target.
        '''
        
        try:
            return self._data[attr].sub("", getattr(target, attr))
        except KeyError:
            return getattr(target, attr)
    
    def mask(self, key, value):
        '''
        Mask the provided text using the regular expression 
        corresponding to the provided key.

        Parameters
        ----------
            key : str
                The key referring to one recorded regular expression.
            
            value : str
                Any text to be masked.
        '''
        
        try:
            return self._data[key].sub("", value)
        except KeyError:
            return value

class DB(object):
    '''
    Store a searchable collection of elements of the same type.
    '''
        
    def __init__(self, values, dbtype=None):
        '''
        Build the database object from a list of object of the same 
        type.

        Parameters
        ----------
            values : list
                A list of Label objects (or child classes). All object
                of this list has to be of the same type.

            dbtype : type
                The type of the object collection. If None, this is 
                guessed from the first object in the values list. 
        '''
        
        # dbtype check
        for attr in ("ID", "text"):
            if not hasattr(dbtype, attr):
                raise TypeError("Element type of the database must have an"
                               f" '{attr}' attribute.")
        
        # type guess
        if dbtype is None:
            self.element_type = type(values[0])
        else:
            self.element_type = dbtype

        # type check
        if not all( type(x) is self.element_type for x in values ):
            raise TypeError("Input values must only be"
                           f" {self.element_type.__name__} objects.")
        
        self._dict = dict( (x.ID, x) for x in values )
        self._ids = list(self._dict.keys())    
    
    @property
    def element_type(self):
        if not hasattr(self, "_element_type"):
            raise ValueError("This database has no defined element type.")
        return self._element_type
    
    @element_type.setter
    def element_type(self, value):
        if hasattr(self, "_element_type"):
            raise ValueError("Element type cannot be defined after object"
                             " initialization.")
        self._element_type = value
    
    def dump_db(self, f):
        '''
        Save the database in JSON format.
        '''

        obj = [ x.export() for x in self ]
        json.dump(obj, f, ensure_ascii=False, indent=4)

    def get(self, value):
        '''
        Get an object of the database by providing with its ID.
        '''

        return self._dict[value]
    
    def is_indexed(self):
        '''
        Returns True if the class method make_index was called.
        '''
        
        return hasattr(self, "_index")
    
    def make_index(self, method=1, min_len=1, keys=None, masks=None, ignore_ids=True):
        '''
        Make a database search index out of the tokens collected from
        all elements of the collection.
        
        Parameters
        ----------
            method : int
                Select to the method with which to build the index.
                Available methods:    
                - 1: Associate tokens with database elements
                - 2: Associate tokens with database elements, and
                a normalized TF-IDF score.
                       
                Default: 1
            
            min_len : int
                Discards tokens with less than the provided character 
                number.
            
            keys : list
                Only fetch tokens from the provided key list. By 
                default, build a database from the value stored in 
                the "text" field.

        '''
        
        # by default, search in all elements of the database items except the 
        # ID.
        if keys is None:
            keys = tuple( key 
                           for key in self.element_type.keys 
                           if not ignore_ids or key != "ID" )
        
        # tokenize pattern
        token_pattern = get_word_tokenize_pattern(min_len)
        
        # index and stored parameters
        self._index = defaultdict(list)
        self._parameters = {"method": method, "token_pattern": token_pattern,
                            "keys": keys, "masks": masks}
        
        # - method 1
        # Associate each unique token with every database item that contains 
        # it. Each item is stored in association with the frequency of the 
        # token in its content.
        if method == 1:
            vectorizer = CountVectorizer(token_pattern=token_pattern, 
                                         strip_accents="unicode")
        
        # - method 2
        # Associate each unique token with every database item that contains 
        # it. Each item is stored in association with the normalized TF-IDF
        # score of the token in its content.
        if method == 2:
            vectorizer = TfidfVectorizer(token_pattern=token_pattern,
                                         strip_accents="unicode")
        
        # Store the matrix containing scores for each unique token in each
        # element of the database, then build the index linking tokens with
        # items and scores.
        corpus = self.get_corpus(keys=keys, masks=masks)
        score_matrix = vectorizer.fit_transform(corpus)
        item_scores = defaultdict(dict)
        for j, token in enumerate(vectorizer.get_feature_names_out().tolist()):
            for i, x in enumerate(self):
                score = score_matrix[i,j]
                if score:
                    self._index[token].append((x, score))
                    item_scores[x.ID][token] = score
        
        # compute maximum scores
        self._max_scores = defaultdict(lambda: 0)
        for x in self:
            for token in self.get_item_tokens(x.ID):
                self._max_scores[x.ID] += item_scores[x.ID][token]
        
    def dump_index(self, fout):
        '''
        Save the index in a JSON formatted file.
        '''
        
        if not self.is_indexed():
            raise ValueError("Database must be indexed with the 'make_index'"
                             " method prior to saving.")

        # Numbers are converted to Python native float, for serialization purpose.
        # It may result in scoring imprecision when using a dumped database.
        index = dict( (token, 
                      [ (x.ID, float(score)) for x, score in self._index[token] ])
                      for token in self._index )
        max_scores = dict( (ID, float(self._max_scores[ID]))
                            for ID in self._max_scores )
        json.dump({"index": index, 
                   "parameters": self._parameters,
                   "max_scores": max_scores}, 
                  fout, ensure_ascii=False, indent=4)
    
    def load_index(self, f):
        '''
        Load an index from a JSON file.
        '''
        
        data = json.load(f)
        self._index = dict( (token,
                             [ (self.get(ID), float(score)) for ID, score in data["index"][token] ])
                             for token in data["index"] )
        self._parameters = data["parameters"]
        self._max_scores = data["max_scores"]

    def get_item_tokens(self, ID):
        '''
        Uses the parameters of the index to generate tokens for a given
        item.

        Parameters
        ----------
            ID : str
                Database item identifier
        '''

        # make a list of keys (use index parameters)
        keys = self._parameters["keys"]
        if keys is None:
            keys = self.element_type.keys
        elif type(keys) is str:
            keys = [keys]
            
        # make a list of masks (use index parameters)
        masks = self._parameters["masks"]
        if masks is None:
            masks = []
        elif type(masks) is Mask:
            masks = [masks]
        
        # identify the tokens
        x = self._dict[ID]
        tokens = []
        pattern = self._parameters["token_pattern"]
        for key in keys:
            s = getattr(x, key)
            for mask in masks:
                s = mask.mask(key, s)
            tokens += regexp_tokenize(strip_accents(s.lower()), pattern)
        return tokens

    def search(self, query, mismatch_rule=mismatch_rule, 
               filtering=lambda x: True, scoring="w"):
        '''
        Search elements of the database with the query text.

        Parameters
        ----------
            query : str
                A text that will be tokenized and search over the 
                database index.

            mismatch_rule : function
                A function that takes the query value as unique 
                argument and returns the regular expression part 
                parametring a fuzzy match.

            filtering : function
                A function that evaluates every matched element and
                returns True or False whether these elements have to be
                kept in the final result of the search.

            scoring : str
                Set up the scoring method.
                    "w" The score is calculated as the product of the 
                    rates of matching token in the query and in the
                    subject, then weighted accounting for mismatches 
                    and TF-IDF scores (default).
                    
                    "l" The score is calculated as the normalized 
                    Levenshtein similarity between the query and the 
                    target. This Levenshtein similarity is calculated
                    on a simplified version of the text, removing 
                    accents, case and treating consecutive white spaces
                    as single space characters.
                    
                    "w+l" The score is calculated as the product of the 
                    two previous methods' results.
                    
        '''
                
        if not self.is_indexed():
            raise ValueError("Database must be indexed with the 'make_index'"
                             " method prior to search")
        
        # extract token from the query
        query_tokens = regexp_tokenize(strip_accents(query.lower()), 
                                       self._parameters["token_pattern"])
        
        # search database tokens with regular expression and score the possible 
        # matches
        matched_tokens = defaultdict(list)

        # build the token search function
        def search_tokens(q, mismatch_rule=mismatch_rule, filtering=filtering):
            return self.get_token_matches(q, mismatch_rule, filtering).items()

        # search every token onto the database index
        for q in query_tokens:

            # x_ID:     ID of the matched element in the database
            # match:
            #   (token:     matched token in this element (not used)
            #    identity:  similarity score (Levenshtein)
            #    score:     TD-IDF score)

            # order matched tokens by matched database item
            for x_ID, match in search_tokens(q):
                matched_tokens[x_ID].append(match)
                
        # score matches while tracking tokens that were matched multiple times
        hit_scoring = defaultdict(lambda: [0, 0])
        for x_ID in matched_tokens:
            subject_tokens = self.get_item_tokens(x_ID)
            for token, identity, score in matched_tokens[x_ID]:

                # consume matched tokens while scoring
                try:
                    subject_tokens.remove(token)

                # if a subject token has already been scored, it means that it
                # was matched by multiple query tokens and therefore needs to be
                # ignored
                except ValueError:
                    continue

                # with the scoring method implying Levenshtein distance, do not 
                # account for the identity as mismatches will be evaluated further
                if scoring != "w":
                    identity = 1

                # the hit score of a given collecting event is the sum of the 
                # normalized TFIDF scores matched in this collecting event any
                # of the query tokens
                hit_scoring[x_ID][0] += score*identity
                
                # count the number of token that matched this collecting event
                hit_scoring[x_ID][1] += 1
                
        # return a list of the matches ordered by normalized score (high to low)
        result = []

        # scoring methods w and w+l includes token scores
        if scoring in ("w", "w+l"):
            for ID, scores in hit_scoring.items():

                score, n = scores
                
                # The score is normalized by the maximum score (i.e. if all 
                # token are matched in the collecting event.
                score /= self._max_scores[ID]
                
                # ...and weighted by the number of matching token.
                score *= (n/len(query_tokens))
                
                # ...and with method w+l, weighted by the normalized 
                # Levenshtein distance
                if scoring == "w+l":
                    score *= (1-get_norm_leven_dist(query, self.get(ID).text, 
                                                    simplify=True))
                result.append((self.get(ID), score))
        elif scoring == "l":
            for ID, scores in hit_scoring.items():
                score, n = scores

                # in method l, the score is the normalized Levenshtein distance
                score = 1-get_norm_leven_dist(query, self.get(ID).text, 
                                              simplify=True)
                result.append((self.get(ID), score))
        else:
            raise ValueError(f"unknown scoring method: {repr(scoring)}")

        # result are sorted from best to lowest matching score    
        result.sort(key=lambda x: x[1], reverse = True)
        return result      
            
    def get_token_matches(self, value, mismatch_rule=mismatch_rule, 
                          filtering=lambda x: True):
        '''
        Find elements with matching tokens, return a list of IDs with 
        the associated token's TF-IDF score.
        
        Parameters
        ----------
            value : str
                The token to be searched.
            
            mismatch_rule : function|None
                If defined as a function, this function will take a 
                single argument, the token, and return a fuzzy regular 
                expression. If defined as None, it will look for exact 
                matches.
        '''
        
        # matched tokens are listed for each database item
        result = defaultdict(list)

        # retrieve matching tokens
        if mismatch_rule is None:
            try:
                for x, score in self._index[value]:
                    if not filtering(x):
                        continue
                    result[x.ID].append((value, 1, score))
            except KeyError:
                pass
        else:
            pattern = regex.compile(fr"(?:{value}){mismatch_rule(value)}")
            
            # list matching tokens in each database item x and associated 
            # scores
            for token, hits in self._index.items():
                m = pattern.fullmatch(token)
                if m is None: continue
                d = levenshtein(token, value)
                l = max((len(token), len(value)))
                identity = 0 if d > l else (1 - d/l)
                for x, score in hits:
                    if not filtering(x): continue
                    result[x.ID].append((token, identity, score))
        
        # for each database item, only the best matched token is kept, ranked
        # by identity, then by TF-IDF score
        return dict(
            (x_ID, sorted(matched_tokens, 
                          key=lambda x: (x[1], x[2]), 
                          reverse=True)[0])
            
            for x_ID, matched_tokens in result.items() )

    def get_corpus(self, keys=None, masks=None, join="\n"):
        '''
        Returns a generator function that yields text values of the 
        database objects.

        Parameters
        ----------
            keys : str|list
                One or a list of attribute names referring to the text 
                values that have to be retrieved from each object of 
                the database. Default: None.
            
            masks : Mask
                One or a list of masks that can be used on retrieved 
                values to remove specific parts of the text. Default: 
                None.
            
            join : str
                If multiple keys were provided, concatenate the 
                corresponding text values with this character string.
        '''
        
        # make a list of keys
        if keys is None:
            keys = self.element_type.keys
        elif type(keys) is str:
            keys = [keys]
            
        # make a list of masks
        if masks is None:
            masks = []
        elif type(masks) is Mask:
            masks = [masks]
        
        # generate the corpus
        for x in self:
            fulltext = []
            for key in keys:
                s = getattr(x, key)
                for mask in masks:
                    s = mask.mask(key, s)
                fulltext.append(s)
            yield join.join(fulltext)
    
    def subset(self, filtering):
        '''
        Return a subsetted database. Each element of the current 
        database is passed as an argument to the filtering function and
        kept only when the function returns True. Existing indexes are 
        removed in the process.
        '''
        
        values = [ x for x in self._dict.values() if filtering(x) ]
        subdb = DB.__new__(DB)
        subdb.__init__(values, dbtype=self.element_type)
        subdb.__name__ = self.__class__.__name__
        return subdb
        
    def __len__(self):
        '''
        Returns the number of elements in the database.
        '''
        
        return len(self._dict)
    
    def __iter__(self):
        '''
        Iterate over elements of the database.
        '''
        
        return iter( self._dict[key] for key in self._dict )

class LabelDB(DB):
    '''
    Store label data and allow text search.
    '''
    
    def __init__(self, values):
        '''
        Build a DB object using exclusively Label objects.
        '''
                
        # type check and DB build
        DB.__init__(self, values, dbtype=Label)
        
class CollectingEventDB(DB):
    '''
    Store collecting events and allow text-based search.
    '''
    
    def __init__(self, values):
        '''
        Build a DB object using exclusively CollectingEvent objects.
        '''
        
        # type check and DB build
        DB.__init__(self, values, dbtype=CollectingEvent)
    
    def has_date_index(self):
        '''
        Returns True if the class method make_date_index has been 
        called.
        '''
        
        return hasattr(self, "_date_index")
    
    def make_date_index(self, **allow_tags):
        '''
        Intepret the date fields of the collecting events to make them 
        searchable with types from the library datetime and the 
        elieclustering.date.DateRange type.

        Parameters
        ----------
            **allow_tags
                Keyword arguments correspondig to valid DatePatternTag 
                can be used here to limit date parsing to specific 
                format or precision level.
        '''

        parser = elieclustering.date.DatePatterns(**allow_tags)
        self._date_index = []
        for x in self:
            date = parser.find_date(x.date)
            
            # If no date could be found, the collecting event is not added to 
            # the index
            if date is None:
                continue
            
            # Otherwise, it is always added as a daterange
            elif type(date) is elieclustering.date.Date:
                daterange = elieclustering.date.DateRange(date, date)
            else:
                daterange = date
            self._date_index.append([daterange, x.ID])
    
    def dump_date_index(self, fout):
        '''
        Store the date index in a JSON formatted file.
        '''
        
        data = [ (daterange.get_isoformat().split(" - "), ID) 
                  for daterange, ID in self._date_index ]
        json.dump(data, fout, ensure_ascii=False, indent=4)

    def load_date_index(self, f):
        '''
        Load the date index from a JSON formatted file.
        '''
        
        self._date_index = []
        for dates, ID in json.load(f):
            daterange = []
            for date in dates:
                date = date.split("-")
                if len(date[0]) == 2: date[0] = f"'{date[0]}"
                daterange.append(elieclustering.date.Date(*date))
            self._date_index.append([elieclustering.date.DateRange(*daterange), ID])
        
    def search_by_date(self, query, assume_same_century=False, **allow_tags):
        '''
        Find collecting events that overlap with the given date or
        daterange.
        '''
        
        if not self.has_date_index():
            raise ValueError("You must build the date index with the"
                             " make_date_index method prior to use the" 
                             " search_by_date method")
        
        if type(query) is str:
            parser = elieclustering.date.DatePatterns(**allow_tags)
            date = parser.find_date(query)
            if date is None:
                raise ValueError(f"{query} was not recognized as a date")
            query = date
        elif type(query) not in (elieclustering.date.Date, elieclustering.date.DateRange):
            raise TypeError(f"Invalid type for the query ({type(query)}),"
                             " should be str, elieclustering.date.Date or"
                             " elieclustering.date.DateRange")
        return [ self.get(ID) 
                  for daterange, ID in self._date_index 
                  if daterange.overlap_with(query, 
                                            assume_same_century) ]
        
# =============================================================================
# FUNCTIONS
# -----------------------------------------------------------------------------
def load_labels(f):
    '''
    Build a label database from data stored in a JSON file.
    '''
    
    return LabelDB([ Label(**x) for x in json.load(f) ])

def load_collecting_events(f):
    '''
    Build a collecting event database from data stored in a JSON file.
    '''

    return CollectingEventDB([ CollectingEvent(**x) for x in json.load(f) ])

def read_googlevision_output(f):
    '''
    Read a Google Vision JSON output and return the full text.
    '''

    for response in json.load(f)["responses"]:
        yield response["fullTextAnnotation"]["text"]

def data_from_googlevision(f, identifier, start=1):
    '''
    Reads transcript from Google Vision JSON output and returns a list
    of dict object with a unique identifier and the associated
    transcript.

    Parameters
    ----------
        f : file
            A file object open in read mode and pointing towards a
            Google Vision JSON output.

        identifier : function
            A function that takes in argument the index of the 
            transcript in the JSON input and returns a str that will be
            used as a unique identifier.
        
        start : int
            Starting index for the identifier function. Default: 1.
    '''
    
    data_list = []
    for text in read_googlevision_output(f):
        data_list.append({
            "ID": identifier(len(data_list)+start),
            "text": text})
    return data_list

def parse_json_db(f):
    '''
    Stream dictionnary objects from the input file object that should 
    contains a JSON label or collecting event database.
    '''

    # each label is comprised within curly brackets, there are no nested
    # brackets
    s, on = "", False
    for line in f:
        start, end = line.find("{"), line.find("}")
        if start == -1:
            if on: 
                s += line[:end]
                if end > -1:
                    yield json.loads("{"+s+"}")
                    s, on = "", False
            elif end > -1:
                raise ValueError("Input format error: nested curly brackets"
                                 " found")
        else:
            if on:
                raise ValueError("Input format error: nested curly brackets"
                                 " found")
            end = line.find("}")
            s += line[start+1:end]
            if end == -1:
                on = True
            else:
                yield json.loads("{"+s+"}")
                s, on = "", False

def parse_labels(f):
    '''
    Stream labels from the input file object that should contains a 
    JSON label database.
    '''

    for x in parse_json_db(f): yield Label(**x)

def parse_collecting_events(f):
    '''
    Stream labels from the input file object that should contains a 
    JSON label database.
    '''

    for x in parse_json_db(f): yield CollectingEvent(**x)
