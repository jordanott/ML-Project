# Environment


### [Environment Manager](https://github.com/jordanott/ML-Project/blob/master/src/environment/env_manager.py)
The ```Environment``` class controls the POMDP. The ```step``` method takes an action *a* given by the agent. It then produces a new state *s<sub>t+1</sub>* and a reward *r<sub>t+1</sub>*.

### [Environment Helper](https://github.com/jordanott/ML-Project/blob/master/src/environment/env_helper.py)
This file contains helper methods for the ```Environment``` class:
* Loading new files
* Cropping images to produce partially observable states
* Determining overlap of words and eye location
