"""The event-based main loop of Blocks."""
import sqlite3
from abc import ABCMeta, abstractproperty
from collections import Mapping, OrderedDict
from numbers import Integral
from operator import methodcaller, itemgetter

import six
from picklable_itertools import imap, groupby
try:
    from pymongo import ASCENDING, MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


class DefaultOrderedDict(OrderedDict):
    r"""A ordered dictionary that supports default values.

    Parameters
    ----------
    default_factor : callable, optional
        A callable that returns the default value when a key is not found.
        If not given, a KeyError will be raised instead.
    \*args
        Interpreted as ``dict``.
    \*\*kwargs
        Interpreted as ``dict``.

    """
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
    r"""The log of training progress.

    A training log stores the training timeline, statistics and other
    auxiliary information. Information is represented as nested dictionary
    of iteration, key values.

    In addition to the set of records of the training progress, a training
    log has a status object whose attributes describe the current state of
    training e.g. whether training has started, whether termination has
    been requested, etc.

    Parameters
    ----------
    backend : {'default', 'mongo', 'sqlite'}
        The log can be stored in a variety of backends. The `default`
        backend stores the values in a nested dictionary (a
        :class:`DefaultOrderedDict` to be precise). This is a simple and
        fast option, but the log is difficult to access for analysis during
        training, and can only be saved as a pickled Python object.

        The `mongo` option uses a MongoDB database server to store the
        logs. The `sqlite` value will use a local SQLite file to store the
        results. Both of these backends will allow you to analyze results
        easily even during training, and will allow you to save multiple
        experiments to the same database.
    status_exclude : list, optional
        A list of status keys that should be excluded from iteration (and
        hence, printing)
    \*\*kwargs
        Additional keyword arguments are passed to the log's backend e.g.
        database configuration. See the relevant classes for details.

    """
    def __init__(self, backend='default', status_exclude=None, **kwargs):
        self.status = TrainingStatus(exclude=status_exclude)
        if backend == 'default':
            self._log = DefaultOrderedDict(dict)
        elif backend == 'mongo':
            self._log = MongoTrainingLog(**kwargs)
        elif backend == 'sqlite':
            self._log = SQLiteLog(**kwargs)
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
    """The status of the training process.

    By default this contains two keys: `iterations_done` and `epochs_done`.

    Parameters
    ----------
    exclude : list, optional
        A list of status keys (strings) that should be excluded when
        iterating over the status items. This can be useful when you want
        to store a particular tidbit of information that should be
        accessible to e.g. extensions, but don't want this to show up in
        the log at every time step.

    Attributes
    ----------
    exclude : list
        See the parameters section.

    """
    def __init__(self, exclude=None):
        self._status = {'iterations_done': 0, 'epochs_done': 0,
                        'epoch_ends': []}
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
    """A training log stored in a MongoDB database."""
    def __init__(self):
        if not PYMONGO_AVAILABLE:
            raise ImportError('pymongo not installed')
        self.client = MongoClient()
        self.db = self.client.blocks_log
        self.entries = self.db.entries

    def __getitem__(self, key):
        return MongoEntry(self, key)

    def __iter__(self):
        return imap(methodcaller('pop', '_id'),
                    self.entries.find(projection=['_id'],
                                      sort=[('_id', ASCENDING)]))

    def __len__(self):
        return self.entries.count()


@six.add_metaclass(ABCMeta)
class DatabaseEntry(Mapping):
    """A helper class for log entries stored in a database.

    Database entries should only be read when explicitly accessed, that is,
    we don't want ``log[0]['foo'] = 'bar'`` to result in ``log[0]`` being
    read from the database. This class defines some helper functions for a
    log entry that behaves this way.

    """
    def __init__(self, log, key):
        self.log = log
        self.key = key

    def __repr__(self):
        return repr(self.entry)

    @abstractproperty
    def entry(self):
        """Should return the entry as a dictionary when requested."""
        pass

    def __getitem__(self, key):
        return self.entry[key]

    def __iter__(self):
        return iter(self.entry)

    def __len__(self):
        return len(self.entry)


class MongoEntry(DatabaseEntry):
    """A single entry in the MongoDB-based log.

    Parameters
    ----------
    log : :class:`MongoTrainingLog`
        The log this entry belongs to.
    key : int
        The key (``iterations_done``) that this entry belongs to.

    Notes
    -----
    Entries are their own objects so that we can read lazily from the
    MongoDB database, so if the user just wants to write new values, it
    won't retrieve the old values.

    """
    @property
    def entry(self):
        if not hasattr(self, '_entry'):
            self._entry = self.log.entries.find_one({'_id': self.key},
                                                    projection={'_id': False})
        return self._entry

    def __setitem__(self, key, value):
        self.log.entries.update_one({'_id': self.key}, {'$set': {key: value}},
                                    upsert=True)


def _grouper_to_dict(grouper):
    """Converts an iterable of SQLite triplet log entries into a dict."""
    _, entries = grouper
    return dict([(key, value) for iteration, key, value in entries])


class SQLiteLog(Mapping):
    def __init__(self):
        self.connection = sqlite3.connect('blocks.db')
        self.cursor = self.connection.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS log '
                            '(iteration INT, key TEXT, value NULL)')
        self.connection.commit()

    def __getitem__(self, key):
        return SQLiteEntry(self, key)

    def __iter__(self):
        entries = self.cursor.execute('SELECT * FROM log')
        return imap(_grouper_to_dict, groupby(entries, key=itemgetter(0)))

    def __len__(self):
        count, = self.cursor.execute('SELECT COUNT(DISTINCT iteration) '
                                     'FROM LOG').fetchone()
        return count


class SQLiteEntry(DatabaseEntry):
    @property
    def entry(self):
        if not hasattr(self, '_entry'):
            entry = self.log.cursor.execute(
                'SELECT * FROM log WHERE iteration = ?', (self.key,))
            self._entry = dict([(key, value)
                                for iteration, key, value in entry])
        return self._entry

    def __setitem__(self, key, value):
        self.log.cursor.execute('INSERT INTO log VALUES (?, ?, ?)',
                                (self.key, key, value))
        self.log.connection.commit()
