# Medical_ChatBot
Developing a medical chatbot with RAG and LLMs

## Dataset: KERAAL (KinesiothErapy and Rehabilitation for Assisted Ambient Living)

The KERAAL Dataset is a medical database of clinical patients carrying out low back-pain rehabilitation exercises. It can be used as a data source for training and evaluating medical chatbots focused on physical rehabilitation.

### Dataset Overview

The KERAAL dataset is designed for human body movement analysis in the context of low-back pain physical rehabilitation. This dataset was acquired during a clinical study where patients performed 3 low-back pain rehabilitation exercises while being coached by a robotic system.

**Key Features:**
- Recorded from 9 healthy subjects and 12 patients suffering from low-back pain
- Annotated by two rehabilitation doctors
- Includes 3D skeleton sequences captured by Kinect
- RGB videos and 2D skeleton data estimated from videos
- Medical expert annotations for:
  - Assessment of correctness
  - Recognition of errors
  - Spatio-temporal localization of errors

### Dataset Content

| Group | Annotation | RGB Videos | Kinect | Openpose/Blazepose | Vicon | Nb Recordings |
|-------|-----------|------------|--------|-------------------|-------|---------------|
| 1a | xml anvil: err label, bodypart, timespan | mp4, 480x360 | tabular | dictionary | NA | 249 |
| 1b | NA | mp4, 480x360 | tabular | dictionary | NA | 1631 |
| 2a | xml anvil: err label, bodypart, timespan | mp4, 480x360 | tabular | dictionary | NA | 51 |
| 2b | NA | mp4, 480x360 | tabular | dictionary | NA | 151 |
| 3 | error label | avi, 960x544 | tabular | dictionary | tabular | 540 |

### Data Types Included

- **RGB Videos**: Anonymized videos in mp4/avi format
- **Kinect Skeleton Data**: 3D positions and orientations of joints in tabular ASCII format
- **OpenPose/BlazePose Skeleton Data**: 2D/3D positions in COCO pose format
- **Vicon Motion Capture Data**: High-precision skeleton sequences
- **Annotations**: XML anvil format with error labels, body parts, and temporal descriptions

### Links and Resources

- **GitHub Repository**: [https://github.com/nguyensmai/KeraalDataset](https://github.com/nguyensmai/KeraalDataset)
- **Dataset Website**: [https://keraal.enstb.org/KeraalDataset.html](https://keraal.enstb.org/KeraalDataset.html)
- **Download**: [http://nguyensmai.free.fr/KeraalDataset.html](http://nguyensmai.free.fr/KeraalDataset.html)

### Citation

If using this dataset, please cite:

```bibtex
@inproceedings{Nguyen2024IJCNN,
    author = {Sao Mai Nguyen and Maxime Devanne and Olivier Remy-Neris and Mathieu Lempereur and Andre Thepaut},
    booktitle = {International Joint Conference on Neural Networks},
    title = {A Medical Low-Back Pain Physical Rehabilitation Database for Human Body Movement Analysis},
    year = {2024}
}
```

### Use Cases for Medical ChatBot

This dataset can be used for:
1. **Exercise Assessment**: Training models to evaluate if rehabilitation exercises are performed correctly
2. **Error Recognition**: Identifying common mistakes patients make during exercises
3. **Feedback Generation**: Generating personalized feedback for patients based on their performance
4. **Rehabilitation Guidance**: Providing context-aware recommendations for low-back pain rehabilitation
