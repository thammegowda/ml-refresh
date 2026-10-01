import re

# tag::pattern[]
PATTERN = re.compile(r"\(literal\)")  # a code listing with \( that must not become math
# end::pattern[]
