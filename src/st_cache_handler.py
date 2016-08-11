"""File to interact with cache folder to isolate cache handling functionality
from main controllers code.

The CacheHandler should only be accessed by controller classes.

"""

#.-------------------.
#|      imports      |
#'-------------------'
import os
import pickle

#.-------------------.
#|    main entry     |
#'-------------------'
class CacheHandler(object):
    """ Class to facilitate interaction with the cache folder. """

    def __init__(self):
        pass

    def pickle_object(self, filename, instance):
        pickle.dump(instance, open(filename, "wb"))

    def unpickle_object(self, filename):
        return pickle.load(open(filename, "rb"))

    def get_folders(self):
        """ Return list of folders within cache. """
        return [folder for folder in os.listdir('cache')
			    if os.path.isdir(os.path.join('cache', folder))]

    def get_subfolders(self, folder):
        """ Return list of subfolders within cache. """
        folder = os.path.join('cache', folder)
        return [subfolder for subfolder in os.listdir(folder)
		        if os.path.isdir(os.path.join(folder, subfolder))]

    def get_extension(self, filename):
        return os.path.splitext(filename)[1][1:]

    def get_filenames(self, folder, subfolder, ext = None):
        """ Return list of filenames within cache. """
        subfolder = os.path.join('cache', folder, subfolder)
        return [filename for filename in os.listdir(subfolder)
		        if (not os.path.isdir(os.path.join(subfolder, filename))) and 
                   (not ext or self.get_extension(filename) == ext)]

    def save_single(self, folder, subfolder, file, instance):
        """ Save the instance at specified location, and delete all other files in same subfolder. """
        if folder not in self.get_folders():
            os.makedirs(os.path.join('cache', folder))
        if subfolder not in self.get_subfolders(folder):
            os.makedirs(os.path.join('cache', folder, subfolder))
        else:
            # cleanup directory before saving new file. TODO: warn user if not empty.
            for file_name in self.get_filenames(folder, subfolder):
                 os.remove(os.path.join('cache', folder, subfolder, file_name))

        location = os.path.join('cache', folder, subfolder, file)
        self.pickle_object(location, instance)
        return location

    def save_df(self, folder, subfolder, file, data_frame):
        """ Save the DataFrame at specified location, without deleting other files in same subfolder. """
        if folder not in self.get_folders():
            os.makedirs(os.path.join('cache', folder))
        if subfolder not in self.get_subfolders(folder):
            os.makedirs(os.path.join('cache', folder, subfolder))

        location = os.path.join('cache', folder, subfolder, file)
        data_frame.to_csv(location)
        return location

    def load_single(self, folder, subfolder):
        """ Unpickle and return the instance inside first file at specified location. """
        if folder not in self.get_folders() or \
            subfolder not in self.get_subfolders(folder) or \
            len(self.get_filenames(folder, subfolder, "trm")) == 0:
            return None

        file = self.get_filenames(folder, subfolder, "trm")[0] # if multiple files, will use first file only
        location = os.path.join('cache', folder, subfolder, file)
        return self.unpickle_object(location)