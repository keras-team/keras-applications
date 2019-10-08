<!--
If your issue is an implementation question, please ask your question on [StackOverflow](http://stackoverflow.com/questions/tagged/keras) or [join the Keras Slack channel](https://keras-slack-autojoin.herokuapp.com/) and ask there instead of filing a GitHub issue.

The following is a list of frequently asked questions.

- `AttributeError: 'NoneType' object has no attribute 'image_data_format'`
  - It is recommended to import with `from keras.applications import model_name` ***not*** `from keras_applications import model_name` because the keras-applications is not a standalone library.
  - Or, you can use the keras-applications directly with [a work-around](https://github.com/keras-team/keras-applications/issues/54#issuecomment-530954413).
- `ImportError: cannot import name 'ResNeXt50'`
  - The latest releases may not include the latest models.
  - If you want to use the bleeding edge version, you can try `pip install -U git+https://github.com/keras-team/keras git+https://github.com/keras-team/keras-applications`.
- Lack of training configuration
  - The keras-applications is designed for inference only, so don't provide training details such as data augmentation (e.g., rotating, shifting), optimization hyperparameters (e.g., lr, decay), and a release number of ImageNet used for training.
  - For such information, you can check the original repositories shown in the table in README.
-->

### Summary

### Environment
- Python version:
- Keras version:
- Keras-applications version:
- Keras backend with version:

### Logs or source codes for reproduction

