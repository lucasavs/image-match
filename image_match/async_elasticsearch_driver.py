from .signature_database_base import SignatureDatabaseBase
from .signature_database_base import normalized_distance
from .signature_database_base import make_record
from .signature_database_base import get_words
from .signature_database_base import words_to_int
from .signature_database_base import max_contrast
from .signature_database_base import normalized_distance
from operator import itemgetter
from datetime import datetime
import numpy as np
from itertools import product
from collections import deque


class SignatureES(SignatureDatabaseBase):
    """Elasticsearch driver for image-match

    """

    def __init__(self, es, index='images', timeout='10s', size=100,
                 *args, **kwargs):
        """Extra setup for Elasticsearch

        Args:
            es (elasticsearch): an instance of the elasticsearch python driver
            index (Optional[string]): a name for the Elasticsearch index (default 'images')
            timeout (Optional[int]): how long to wait on an Elasticsearch query, in seconds (default 10)
            size (Optional[int]): maximum number of Elasticsearch results (default 100)
            *args (Optional): Variable length argument list to pass to base constructor
            **kwargs (Optional): Arbitrary keyword arguments to pass to base constructor

        Examples:
            >>> from elasticsearch import Elasticsearch
            >>> from image_match.elasticsearch_driver import SignatureES
            >>> es = Elasticsearch()
            >>> ses = SignatureES(es)
            >>> ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            >>> ses.search_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            [
             {'dist': 0.0,
              'id': u'AVM37nMg0osmmAxpPvx6',
              'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
              'score': 0.28797293}
            ]

        """
        self.es = es
        self.index = index
        self.timeout = timeout
        self.size = size

        super(SignatureES, self).__init__(*args, **kwargs)

    async def search_single_record(self, rec, pre_filter=None):
        path = rec.pop('path')
        signature = rec.pop('signature')
        if 'metadata' in rec:
            rec.pop('metadata')

        # build the 'should' list
        should = [{'term': {word: rec[word]}} for word in rec]
        body = {
            'query': {
                   'bool': {'should': should}
            },
            '_source': {'excludes': ['simple_word_*']}
        }

        if pre_filter is not None:
            body['query']['bool']['filter'] = pre_filter

        res = (await self.es.search(index=self.index,
                              body=body,
                              size=self.size,
                              timeout=self.timeout))['hits']['hits']

        sigs = np.array([x['_source']['signature'] for x in res])

        if sigs.size == 0:
            return []

        dists = normalized_distance(sigs, np.array(signature))

        formatted_res = [{'id': x['_id'],
                          'score': x['_score'],
                          'metadata': x['_source'].get('metadata'),
                          'path': x['_source'].get('url', x['_source'].get('path'))}
                         for x in res]

        for i, row in enumerate(formatted_res):
            row['dist'] = dists[i]
        formatted_res = filter(lambda y: y['dist'] < self.distance_cutoff, formatted_res)

        print(formatted_res)

        return formatted_res

    async def insert_single_record(self, rec, refresh_after=False):
        rec['timestamp'] = datetime.now()
        await self.es.index(index=self.index, body=rec, refresh=refresh_after)

    async def delete_duplicates(self, path):
        """Delete all but one entries in elasticsearch whose `path` value is equivalent to that of path.
        Args:
            path (string): path value to compare to those in the elastic search
        """
        matching_paths = [item['_id'] for item in
                          await (self.es.search(body={'query':
                                               {'match':
                                                {'path': path}
                                               }
                                              },
                                         index=self.index))['hits']['hits']
                          if item['_source']['path'] == path]
        if len(matching_paths) > 0:
            for id_tag in matching_paths[1:]:
                await self.es.delete(index=self.index, id=id_tag)

    async def add_image(self, path, img=None, bytestream=False, metadata=None, refresh_after=False):
        rec = make_record(path, self.gis, self.k, self.N, img=img, bytestream=bytestream, metadata=metadata)
        await self.insert_single_record(rec, refresh_after=refresh_after)
    
    async def search_image(self, path, all_orientations=False, bytestream=False, pre_filter=None):
        img = self.gis.preprocess_image(path, bytestream)

        if all_orientations:
            # initialize an iterator of composed transformations
            inversions = [lambda x: x, lambda x: -x]

            mirrors = [lambda x: x, np.fliplr]

            # an ugly solution for function composition
            rotations = [lambda x: x,
                         np.rot90,
                         lambda x: np.rot90(x, 2),
                         lambda x: np.rot90(x, 3)]

            # cartesian product of all possible orientations
            orientations = product(inversions, rotations, mirrors)

        else:
            # otherwise just use the identity transformation
            orientations = [lambda x: x]

        # try for every possible combination of transformations; if all_orientations=False,
        # this will only take one iteration
        result = []

        orientations = set(np.ravel(list(orientations)))
        for transform in orientations:
            # compose all functions and apply on signature
            transformed_img = transform(img)

            # generate the signature
            transformed_record = make_record(transformed_img, self.gis, self.k, self.N)

            l = await self.search_single_record(transformed_record, pre_filter=pre_filter)
            
            result.extend(l)

        ids = set()
        unique = []
        for item in result:
            if item['id'] not in ids:
                unique.append(item)
                ids.add(item['id'])

        r = sorted(unique, key=itemgetter('dist'))
        return r
