Self driving car lane and path detection
=========================================
## Demo
<div align="center">
      <a href="https://www.youtube.com/watch?v=UFQQbTYH9hI-Y">
     <img 
      src="https://i.ytimg.com/vi/UFQQbTYH9hI/maxresdefault.jpg" 
      alt="Demo" 
      style="width:100%;">
      </a>
    </div>

## How to install

To be able to run this, I recommend using Python 3.6 or up.

1. Install the requirements 

```
pip3 install -r requirements.txt
```
This will install all the necessary dependencies for running this. 

2. Download the sample data

The sample data can be downloaded from [here.](https://drive.google.com/file/d/1hP-v8lLn1g1jEaJUBYJhv1mEb32hkMvG/view?usp=sharing) More data will be added soon. 

3. Run the program

``` 
python3 main.py <path-to-sample-data-hevc> 
```

## What's next ? 

- Traffic light, Car, Truck, Bicycle, Motorcycle, Pedestrians and Stop sign detection using YOLOv3.
- Real Time semantic segmentation, or almost real time.
- Fast SLAM.

## Related research

[Learning a driving simulator](https://arxiv.org/abs/1608.01230)

## Credits

[comma.ai for supercombo model](https://github.com/commaai/openpilot/blob/master/models/supercombo.keras)

[Harald Schafer for parts of the code](https://github.com/haraldschafer)

[lanes_image_space.py Code by @Shane](https://github.com/ShaneSmiskol)
