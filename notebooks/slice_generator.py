import numpy as np
import glob
import xarray as xr
from collections.abc import Generator

class slice_generator(Generator):
    """
    A generator that returns a tuple (input_images, output_images), where
    input_images is of shape (1, slice_size, len(vars_), pixels_x, pixels_y)
    output_images is the same shape but starts immediately after end of input
    """
    def __init__(self, img_dir:str, slice_size:int, vars_:list,
                 proc_type:str, pixels_x:int, pixels_y:int, debug:bool):
        self.slice_size = slice_size
        self.vars_ = vars_
        self.proc_type = proc_type
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        
        self.netcdf_dirs = sorted(glob.glob(img_dir+"/*.nc"))
        
        self._file_index = 0
        self._counter = 0
        self._debug = debug
        
        self._dataset_current = xr.open_dataset(self.netcdf_dirs[self._file_index])
        self._dataset_current = self._dataset_current[self.vars_]
        
        # Convert 2m temperature from Kelvin to deg C
        if 't2m' in self.vars_:
            self._dataset_current['t2m'] = self._dataset_current['t2m'] - 273.15
        
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration
    
    def send(self, ignored_args):
        input_images = self._get_slices(start=self._counter,
                                  end=self._counter+self.slice_size,
                                  )
        
        # When we get to the end of the file, open up the next file and pull the ouputs from there, 
        # then get ready for the next iteration
        
        if self._counter + 2*self.slice_size > self._dataset_current.sizes['time']:
            self._counter = 0
            self._file_index += 1
            # loop back to beginning of file list if we get to the end
            if self._file_index == len(self.netcdf_dirs):
                self._file_index = 0
            
            # Open next .nc file in netcdf_dirs
            self._dataset_current = xr.open_dataset(self.netcdf_dirs[self._file_index])
            self._dataset_current = self._dataset_current[self.vars_]

            output_images = self._get_slices(start=self._counter,
                                       end=self._counter + self.slice_size, 
                                       )
            self._counter -= self.slice_size
        
        else:
            output_images = self. _get_slices(start=self._counter + self.slice_size,
                                       end=self._counter + 2*self.slice_size,
                                      )
        
        self._counter += self.slice_size
        return ([input_images, output_images], output_images)

    def _get_slices(self,start:int, end:int):
        if self._debug == True:
            array = self._dataset_current.isel(time=slice(start, end))
            return array
        array = self._dataset_current.isel(time=slice(start, end)).to_array().values
        if self.proc_type == "convlstm":
            # switch first and second axes (in practice from (channels, frames, pixels_x, pixels_y) 
            # to (frames, channels, pixels_x, pixels_y) )
            array = np.moveaxis(array, 0, 1)
            # add empty dimension in front for ConvLSTMs
            array = array.reshape(-1, self.slice_size, len(self.vars_), self.pixels_x, self.pixels_y)
            return array
        else:
            raise NameError("proc_type is not recognized.")