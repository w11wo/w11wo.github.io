---
title: "Seeing Is Locating: Spatial Awareness and Geographic Bias in Decision Language Models"
collection: publications
permalink: /publication/2026-11-23-geodecisions/
excerpt: "Decision models, which output probability distributions over candidate labels instead of generating text, are increasingly used as classifiers and evaluators for inputs associated with real-world locations. However, it remains unclear whether they can leverage spatial context and whether their predictions are geographically biased. We introduce GeoDecisions, a multimodal benchmark for evaluating the spatial awareness and geographic bias of decision models. GeoDecisions comprises two tasks: (1) geo-aware image classification, and (2) location rating on objective and subjective topics. Evaluating six decision models against supervised task-specific models, we find that spatial awareness requires vision: textual coordinates alone provide insufficient context for classifying everyday photos and satellite images. We also find substantial geographic bias. On everyday photos, despite achieving higher accuracy, open-weight decision models exhibit more spatially clustered errors than supervised models. Moreover, across all five subjective topics, every decision model gives lower ratings to locations with lower socioeconomic conditions. Finally, we propose a training-free method that combines decision models with a location prior to improve their spatial awareness without substantially increasing performance bias. Our code is available at https://github.com/cruiseresearchgroup/GeoDecisions."
date: 2026-11-23
venue: 'arXiv preprint arXiv:2611.30564'
paperurl: 'https://arxiv.org/abs/2611.30564'
citation: 'Wongso, W., Muhammad, H. M., Lin, X., Xue, H., & Salim, F. D. (2026). Seeing Is Locating: Spatial Awareness and Geographic Bias in Decision Language Models. <i>arXiv preprint arXiv:2611.30564.</i>.'
---

## Abstract

Decision models, which output probability distributions over candidate labels instead of generating text, are increasingly used as classifiers and evaluators for inputs associated with real-world locations. However, it remains unclear whether they can leverage spatial context and whether their predictions are geographically biased. We introduce GeoDecisions, a multimodal benchmark for evaluating the spatial awareness and geographic bias of decision models. GeoDecisions comprises two tasks: (1) geo-aware image classification, and (2) location rating on objective and subjective topics. Evaluating six decision models against supervised task-specific models, we find that spatial awareness requires vision: textual coordinates alone provide insufficient context for classifying everyday photos and satellite images. We also find substantial geographic bias. On everyday photos, despite achieving higher accuracy, open-weight decision models exhibit more spatially clustered errors than supervised models. Moreover, across all five subjective topics, every decision model gives lower ratings to locations with lower socioeconomic conditions. Finally, we propose a training-free method that combines decision models with a location prior to improve their spatial awareness without substantially increasing performance bias. Our code is available at [https://github.com/cruiseresearchgroup/GeoDecisions](https://github.com/cruiseresearchgroup/GeoDecisions).

## BibTeX Citation

```bibtex
@misc{wongso2026seeinglocatingspatial,
  title         = {Seeing Is Locating: Spatial Awareness and Geographic Bias in Decision Language Models},
  author        = {Wilson Wongso and Hada Melino Muhammad and Xiachong Lin and Hao Xue and Flora D. Salim},
  year          = {2026},
  eprint        = {2611.30564},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CV},
  url           = {https://arxiv.org/abs/2611.30564}
}
```