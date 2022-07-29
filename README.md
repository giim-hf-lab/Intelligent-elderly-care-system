# Intelligent-elderly-care-system
**Introduction**

This is a repository platform for elderly abnormal behavior care, which can detect the abnormal behaviors including cough, headache, chestpain, backpain,fall,sit down and stand up. Combining with sit down and stand up can determine the sedentary time of elder. In addition, this platform provides the function of recording abnormal information, such as the moment when the behavior occurs, the duration, etc. It should be noted that the repository is a showcase for demonstrations and research. You can run this real-time platform acccording to the following preparation:

**Configure**

In hardware: Windows10-64 bit, NVIDIA GTX2060, CUDA10.1

In sofware: Anaconda3, Python3.7

**Run**

There are other open source library shoud be installed, you can install these as the command in your compputer terminal:

``pip install -r requirement.txt``

Then, download the weights file of the model from google cloud according to the link below, and put it into folder"/xx/works_dir/":
``https://drive.google.com/file/d/1edWlk2reHC1gqkuNzVfbMR4pkDX1G7MS/view?usp=sharing``

Last, run the py file:

``python demo.py``





