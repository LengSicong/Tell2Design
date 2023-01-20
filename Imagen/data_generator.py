import os
import numpy as np
import PIL
import sys

from PIL import Image

from gan_utils import txt_to_onehot, \
    txt_from_onehot, \
    get_images, \
    get_vocab, \
    load_image


class DataGenerator():
    def __init__(self, images, txts, vocab,
                 to_fit=True, batch_size=32, dim=(256, 256),
                 n_channels=3, shuffle=True, transform=None,
                 normalize=True, ret_filenames=False,
                 channels_first=False,
                 tag_transform=None,
                 silent=True,
                 as_array=False,
                 resize=False,
                 limit=None,
                 return_raw_txt=False,
                 use_text_encodings=False):

        self.images = images
        self.txts = txts
        self.vocab = vocab
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.ret_filenames = ret_filenames
        self.channels_first = channels_first
        self.ndim = 1
        self.as_array = as_array
        self.resize = resize
        self.limit = limit
        self.return_raw_txt = return_raw_txt
        self.use_text_encodings = use_text_encodings
        self.on_epoch_end()


        self.transform = transform
        self.tag_transform = tag_transform
        self.silent = silent

        if self.transform is None:
            self.transform = lambda x: x
        else:
            # cannot have a transform and normalize
            normalize = False

        self.normalize = normalize

        self.txts_oh = {}

        self._preload_txts()

    def _preload_txts(self):
        print("preloading txt files...")

        tot = len(self.images)
        count = 0

        txts = {}

        for txt in self.txts:
            bn = os.path.splitext(os.path.basename(txt))[0]
            txts[bn] = txt

        for img in self.images:
            count += 1
            print("\rprocessing {}/{}".format(count, tot), end='', flush=True)

            bn = os.path.basename(img)
            bn = os.path.splitext(bn)[0]

            try:
                txt = txts[bn]

                if self.use_text_encodings:
                    oh = np.load(txt)
                else:
                    with open(txt, 'r') as f:
                        try:
                            txt_data = f.read()
                        except Exception as ex:
                            print(ex)
                            print(f"with file {txt}")

                            txt_data = ""

                    if self.tag_transform is not None:
                        txt_data = self.tag_transform(txt_data)

                    if not self.return_raw_txt:
                        oh = txt_to_onehot(self.vocab, txt_data)
                    else:
                        oh = txt_data

                self.txts_oh[bn] = oh

            except KeyError:
                continue

        print("\ndone preloading txt onehots")

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """

        if self.limit:
            return (int(min(np.floor(len(self.images) / self.batch_size), self.limit / self.batch_size)))

        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # print(f"getting index: {index}")
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.images[k] for k in indexes]

        # Generate data
        return self._generate_X(list_IDs_temp)

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self.indexes = np.arange(len(self.images))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        fn = []
        if self.channels_first:
            if self.batch_size > 1:
                X = np.zeros((self.batch_size, self.n_channels, *self.dim))
            else:
                X = np.zeros((self.n_channels, *self.dim))
        else:
            X = np.zeros((self.batch_size, *self.dim, self.n_channels))

        y = np.zeros((self.batch_size, len(self.vocab)))

        # Generate data
        for i, img in enumerate(list_IDs_temp):
            # Store sample

            bn = os.path.basename(img)
            bn = os.path.splitext(bn)[0]

            if bn not in self.txts_oh:
                if not self.silent:
                    print(f"could not find {bn} in preloaded txts")
                continue

            try:
                if self.normalize:
                    the_img = load_image(img, self.dim,
                                         normalize=self.normalize)
                else:
                    the_img = Image.open(img).convert("RGB")

                if self.resize:
                    the_img = the_img.resize(self.dim, Image.BICUBIC)

                the_img = self.transform(the_img)

                if self.as_array:
                    the_img = np.array(the_img)

            except ValueError as v:
                print(f"error processing {img}: {v}")
                self.images.remove(img)
                self.on_epoch_end()
                os.remove(img)

                continue

            except FileNotFoundError as er:
                print(f"error processing {img}: {er}")
                self.images.remove(img)
                self.on_epoch_end()

                continue

            fn.append(img)
            X[i, ] = the_img
            # except Exception as ex:
            #    print("failzors")
            #    print(ex)
            #    continue

            muh_oh = self.txts_oh[bn]
            y[i, ] = muh_oh
            # print(txt_from_onehot(self.vocab, self.txts_oh[bn]))

            # print(f"loading {img}")

        if self.ret_filenames:
            return fn, X, y

        return X, y


class ImageLabelDataset():
    def __init__(self, images, txts, vocab,
                 styles=None,
                 cond_images=None,
                 to_fit=True, 
                 dim=(256, 256),
                 n_channels=3,
                 shuffle=True,
                 transform=None,
                 alt_transform=None, # transform for style and cond images
                 normalize=True,
                 ret_filenames=False,
                 channels_first=False,
                 tag_transform=None,
                 silent=True,
                 as_array=False,
                 resize=False,
                 limit=None,
                 return_raw_txt=False,
                 no_preload=False,
                 use_text_encodings=False):

        self.images = images
        self.txts = txts
        self.vocab = vocab
        self.styles = styles
        self.cond_images = cond_images
        self.has_style = True if styles is not None else False
        self.has_cond = True if cond_images is not None else False
        self.to_fit = to_fit
        self.batch_size = 1
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.ret_filenames = ret_filenames
        self.channels_first = channels_first
        self.ndim = 1
        self.as_array = as_array
        self.resize = resize
        self.limit = limit
        self.return_raw_txt = return_raw_txt
        self.use_text_encodings = use_text_encodings
        self.on_epoch_end()

        self.transform = transform
        self.alt_transform = alt_transform

        if self.alt_transform is None:
            self.alt_transform = self.transform

        self.tag_transform = tag_transform
        self.silent = silent

        if self.transform is None:
            self.transform = lambda x: x
        else:
            # cannot have a transform and normalize
            normalize = False

        self.normalize = normalize

        self.txts_oh = {}
        self.styles_preload = {}
        self.cond_preload = {}

        if not no_preload:
            self._preload_txts()
            if self.has_style:
                self._preload_poses(dest=self.styles_preload,
                                    source=self.styles)
            if self.has_cond:
                self._preload_poses(dest=self.cond_preload,
                                    source=self.cond_images)

    def _preload_txts(self, images=None):
        # print("preloading txt files...")

        if images is None:
            images = self.images

        tot = len(images)
        count = 0

        txts = {}

        for txt in self.txts:
            bn = os.path.splitext(os.path.basename(txt))[0]
            txts[bn] = txt

        for img in images:
            count += 1
            # print("\rprocessing {}/{}".format(count, tot), end='', flush=True)

            bn = os.path.basename(img)
            bn = os.path.splitext(bn)[0]

            try:
                txt = txts[bn]
                if self.use_text_encodings:
                    with np.load(txt) as data:
                        oh = data['arr_0']
                        # print(oh)
                else:
                    with open(txt, 'r', encoding="utf-8") as f:
                        try:
                            txt_data = f.read()
                        except Exception as ex:
                            print(ex)
                            print(f"Error reading text file: {txt}")

                            txt_data = ""

                    if self.tag_transform is not None:
                        txt_data = self.tag_transform(txt_data)

                    if not self.return_raw_txt:
                        oh = txt_to_onehot(self.vocab, txt_data)
                    else:
                        oh = txt_data

                self.txts_oh[bn] = oh

            except KeyError:
                continue

    def _preload_poses(self, dest, source, images=None):
        # print("preloading pose files...")

        if images is None:
            images = self.images

        tot = len(images)
        count = 0

        poses = {}

        for pose in source:
            bn = os.path.basename(pose)

            bn = os.path.splitext(bn)[0]

            poses[bn] = pose

        for img in images:
            count += 1
            # print("\rprocessing {}/{}".format(count, tot), end='', flush=True)

            bn = os.path.basename(img)
            bn = os.path.splitext(bn)[0]

            try:
                pose = poses[bn]
                dest[bn] = pose
            except KeyError:
                continue

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """

        if self.limit:
            return (int(min(np.floor(len(self.images) / self.batch_size), self.limit / self.batch_size)))

        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # print(f"getting index: {index}")
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.images[k] for k in indexes]

        # Generate data
        return self._generate_X(list_IDs_temp)

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self.indexes = np.arange(len(self.images))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        img = list_IDs_temp[-1]
        fn = []
        bn = os.path.basename(img)
        bn = os.path.splitext(bn)[0]

        # if bn not in self.txts_oh:
        #     if not self.silent:
        #         print(f"could not find {bn} in preloaded txts")
        #     print(f"could not find {bn} in preloaded txts")
        #     return

        try:
            if self.normalize:
                the_img = load_image(img, self.dim,
                                     normalize=self.normalize)
            else:
                the_img = Image.open(img).convert("RGB")

            if self.resize:
                the_img = the_img.resize(self.dim, Image.BICUBIC)

            the_img = self.transform(the_img)

            if self.as_array:
                the_img = np.array(the_img)

        except ValueError as v:
            print(f"error processing {img}: {v}")
            self.images.remove(img)
            self.on_epoch_end()
            os.remove(img)

            return

        except FileNotFoundError as er:
            print(f"error processing {img}: {er}")
            self.images.remove(img)
            self.on_epoch_end()

            return

        except Image.DecompressionBombError as v:
            print(f"error processing {img}: {v}")
            self.images.remove(img)
            self.on_epoch_end()
            os.remove(img)

            return

        except PIL.UnidentifiedImageError as v:
            print(f"error processing {img}: {v}")
            self.images.remove(img)
            self.on_epoch_end()
            os.remove(img)

            return
        except OSError as v:
            print(f"error processing {img}: {v}")
            self.images.remove(img)
            self.on_epoch_end()
            os.remove(img)

            return

        fn.append(img)
        X = the_img
        # except Exception as ex:
        #    print("failzors")
        #    print(ex)
        #    continue

        # print(bn)
        muh_oh = self.txts_oh.get(bn, None)

        if muh_oh is None:
            self._preload_txts(images=[img])
            try:
                muh_oh = self.txts_oh[bn]
            except KeyError:
                os.remove(img)

        y = muh_oh
        # print(txt_from_onehot(self.vocab, self.txts_oh[bn]))

        # print(f"loading {img}")

        ret_tup = [X, y]

        if self.has_style:
            pose = self.styles_preload.get(bn, None)

            if pose is None:
                self._preload_poses(dest=self.styles_preload, source=self.styles, images=[img])
                pose = self.styles_preload[bn]

            the_style = Image.open(pose).convert("RGB")
            the_style = self.alt_transform(the_style)

            ret_tup.append(the_style)

        if self.has_cond:
            pose = self.cond_preload.get(bn, None)

            if pose is None:
                self._preload_poses(dest=self.cond_preload,
                                    source=self.cond_images,
                                    images=[img])
                pose = self.cond_preload[bn]

            the_cond = Image.open(pose).convert("RGB")
            the_cond = self.alt_transform(the_cond)

            ret_tup.append(the_cond)

        if self.ret_filenames:
            ret_tup = [fn, *ret_tup]

        # print(X.size())
        # print(y.size())
        return ret_tup
