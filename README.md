# Weather prediction using deep learning

I'd like to train a deep learning model on gridded meteorological data and use it to predict weather in a localized area.

The current state of the art in weather prediction is to use large, computationally intensive numerical models like [WRF](https://www.mmm.ucar.edu/weather-research-and-forecasting-model). These models, however, cannot resolve every interaction (small wind gusts, radiative transfer, cloud microphysics) without taking until the end of days, so instead a *parameterization* approximation is used to model the interaction processes on a larger scale.

## The challenge, simplified:

- Weather is predicted using numerical models like WRF

- WRF resolution usually is too coarse to model processes like cloud microphyics, radiative transfer, etc.

- Parametrizations are used as approximations for these fine-res processes or else the model would take forever

## The answer, simplified:

- Use deep learning, because neural networks *make no assumptions* about underlying data and are also *robust to noise*

- Train NN on daily avg of temp + some other variables, on relatively localized area