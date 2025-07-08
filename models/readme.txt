People-R-People dataset for Darknet/YOLO, v1.0
https://www.ccoderun.ca/programming/2025-01-20_People-R-People/

This dataset is a combination of several datasets.  The purpose is to detect people easily, including difficult crowd conditions.

Images and annotations include:

- https://www.crowdhuman.org/ (both images and annotations were used)
- https://cocodataset.org/ (images without people to use as negative samples)
- https://cocodataset.org/ (images from class #0, limited to between 1 and 3 people max were re-annotated)
- https://github.com/alex000kim/nsfw_data_scraper (only safe-for-work images were used and re-annotated)

Example command you can run to test the weights:

	darknet_02_display_annotated_images people-r-people sample_images/

Darknet/YOLO discord:  https://discord.gg/zSq8rtW

Stéphane Charette, stephanecharette@gmail.com
2025-01-20
