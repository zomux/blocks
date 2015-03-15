"""The event-based main loop of Blocks."""
from collections import Mapping, OrderedDict
from numbers import Integral
from operator import methodcaller

from pymongo import ASCENDING, MongoClient
from six.moves import map


class DefaultOrderedDict(OrderedDict):
    def __init__(self, default_factory=None, *args, **kwargs):
        if default_factory is not None and not callable(default_factory):
            raise TypeError('first argument must be callable')
        self.default_factory = default_factory
        super(DefaultOrderedDict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        args = (self.default_factory,)
        return self.__class__, args, None, None, self.iteritems()


class TrainingLog(Mapping):
    def __init__(self, backend='default', **kwargs):
        self.status = TrainingStatus()
        if backend == 'default':
            self._log = DefaultOrderedDict(dict)
        elif backend == 'mongo':
            self._log = MongoTrainingLog(**kwargs)
        else:
            raise ValueError('unknown backend')

    def __getitem__(self, key):
        if not isinstance(key, Integral) or key < 0:
            raise TypeError('invalid timestamp: {}'.format(key))
        return self._log[key]

    def __iter__(self):
        return iter(self._log)

    def __len__(self):
        return len(self._log)

    @property
    def current_entry(self):
        return self[self.status['iterations_done']]

    @property
    def previous_entry(self):
        return self[self.status['iterations_done'] - 1]


class TrainingStatus(Mapping):
    def __init__(self, exclude=None):
        self._status = {'iterations_done': 0, 'epochs_done': 0}
        self.exclude = [] if exclude is None else exclude

    def __getitem__(self, key):
        return self._status[key]

    def __setitem__(self, key, value):
        self._status[key] = value

    def __iter__(self):
        return (key for key in self._status if key not in self.exclude)

    def __len__(self):
        return len(self._status)


class MongoTrainingLog(Mapping):
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client.blocks_log
        self.entries = self.db.entries

    def __getitem__(self, key):
        return MongoEntry(self, key)

    def __iter__(self):
        return map(methodcaller('pop', '_id'),
                   self.entries.find(projection=['_id'],
                                     sort=[('_id', ASCENDING)]))

    def __len__(self):
        return self.entries.count()


class MongoEntry(Mapping):
    def __init__(self, log, key):
        self.log = log
        self.key = key

    @property
    def entry(self):
        if not hasattr(self, '_entry'):
            self._entry = self.log.entries.find_one({'_id': self.key},
                                                    projection={'_id': False})
        return self._entry

    def __getitem__(self, key):
        return self.entry[key]

    def __iter__(self):
        return iter(self.entry)

    def __len__(self):
        return len(self.entry)

    def __setitem__(self, key, value):
        self.log.entries.update_one({'_id': self.key}, {'$set': {key: value}},
                                    upsert=True)
