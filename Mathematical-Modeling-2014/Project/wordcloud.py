#test of pytagcloud

from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts

YOUR_TEXT = "A tag cloud is a visual representation for text data, typically\
used to depict keyword metadata on websites, or to visualize free form text."

tags = make_tags(get_tag_counts(YOUR_TEXT), maxsize=120)

create_tag_image(tags, 'cloud_large.png', size=(900, 600), fontname='Lobster')

