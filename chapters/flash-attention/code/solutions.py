"""Solution snippets for online softmax and FlashAttention."""
from scratch.flash_attention import attention_memory_elements


# tag::memory-example[]
def memory_elements_for_4096():
    return attention_memory_elements(4096)
# end::memory-example[]
