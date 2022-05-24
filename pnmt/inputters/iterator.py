"""Contains all methods relate to iteration."""
import torchtext.data
from torchtext.data.batch import Batch
from pnmt.utils.logging import logger
import pdb

class Batch(Batch):
    def __init__(self, data=None, dataset=None, device=None, pre_train_tokenizer=None):
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names
            input_field = []
            target_field = []
            for k, v in dataset.fields.items():
                if v is not None:
                    if 'is_target' in dir(v):
                        if v.is_target:
                            target_field.append(k)
                        else:
                            input_field.append(k)
            self.input_fields=input_field
            self.target_fields=target_field
            #pdb.set_trace()
            for (name, field) in dataset.fields.items():
                if field is not None and name!='tokenizer':
                    batch = [getattr(x, name) for x in data]
                    if name == 'src':
                        use_pre_trained_model_for_encoder = field.base_field.pre_train_tokenizer
                    else:
                        use_pre_trained_model_for_encoder = False
                    setattr(self, name, field.process(batch,
                                                     device=device,
                                                     use_pre_trained_model_for_encoder=use_pre_trained_model_for_encoder,
                                                     pre_train_tokenizer=pre_train_tokenizer)
                            )




def batch_iter(data, batch_size, batch_size_fn=None, batch_size_multiple=1):
    """Yield elements from data in chunks of batch_size, where each chunk size
    is a multiple of batch_size_multiple.

    This is an extended version of torchtext.data.batch.
    """
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far >= batch_size:
            overflowed = 0
            if size_so_far > batch_size:
                overflowed += 1
            if batch_size_multiple > 1:
                overflowed += (
                    (len(minibatch) - overflowed) % batch_size_multiple)
            if overflowed == 0:
                yield minibatch
                minibatch, size_so_far = [], 0
            else:
                if overflowed == len(minibatch):
                    logger.warning(
                        "The batch will be filled until we reach %d,"
                        "its size may exceed %d tokens"
                        % (batch_size_multiple, batch_size)
                        )
                else:
                    yield minibatch[:-overflowed]
                    minibatch = minibatch[-overflowed:]
                    size_so_far = 0
                    for i, ex in enumerate(minibatch):
                        size_so_far = batch_size_fn(ex, i + 1, size_so_far)
    if minibatch:
        yield minibatch


def _pool(data, batch_size, batch_size_fn, batch_size_multiple,
          sort_key, random_shuffler, pool_factor, pre_train_tokenizer):
    for p in torchtext.data.batch(
            data, batch_size * pool_factor,
            batch_size_fn=batch_size_fn):
        p_batch = list(batch_iter(
            sorted(p, key=sort_key),
            batch_size,
            batch_size_fn=batch_size_fn,
            batch_size_multiple=batch_size_multiple))
        for b in random_shuffler(p_batch):
            yield b


class OrderedIterator(torchtext.data.Iterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 pool_factor=1,
                 batch_size_multiple=1,
                 yield_raw_example=False,
                 pre_train_tokenizer=None,
                 **kwargs):
        super(OrderedIterator, self).__init__(dataset, batch_size, **kwargs)
        self.batch_size_multiple = batch_size_multiple
        self.yield_raw_example = yield_raw_example
        self.dataset = dataset
        self.pool_factor = pool_factor
        self.pre_train_tokenizer = pre_train_tokenizer

    def create_batches(self):
        if self.train:
            if self.yield_raw_example:
                self.batches = batch_iter(
                    self.data(),
                    1,
                    batch_size_fn=None,
                    batch_size_multiple=1)
            else:
                self.batches = _pool(
                    self.data(),
                    self.batch_size,
                    self.batch_size_fn,
                    self.batch_size_multiple,
                    self.sort_key,
                    self.random_shuffler,
                    self.pool_factor,
                    self.pre_train_tokenizer)
        else:
            self.batches = []
            for b in batch_iter(
                    self.data(),
                    self.batch_size,
                    batch_size_fn=self.batch_size_fn,
                    batch_size_multiple=self.batch_size_multiple):
                self.batches.append(sorted(b, key=self.sort_key))

    def __iter__(self):
        """
        Extended version of the definition in torchtext.data.Iterator.
        Added yield_raw_example behaviour to yield a torchtext.data.Example
        instead of a torchtext.data.Batch object.
        """
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a
                    # minibatch be sorted by decreasing order, which
                    #  requires reversing relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                if self.yield_raw_example:
                    yield minibatch[0]
                else:
                    yield Batch(
                        minibatch,
                        self.dataset,
                        self.device,
                        self.pre_train_tokenizer
                    )
            if not self.repeat:
                return


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch, max_tgt_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt[0]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
