import asyncio
import math
import os
import numpy as np

from PIL import Image


def is_video(uri):
    vid_exts = [".webm", ".mp4", ".mkv", ".avi", ".gif"]
    ext = os.path.splitext(uri)[1]
    for vid_ext in vid_exts:
        if ext == vid_ext:
            return True

    return False


def is_image(uri):
    vid_exts = [".png", ".jpg", ".jpeg", ".webp"]
    ext = os.path.splitext(uri)[1]
    for vid_ext in vid_exts:
        if ext == vid_ext:
            return True

    return False


async def split_animation(url, output, fps=30):
    fn = os.path.basename(url)
    image_id = os.path.splitext(fn)[0]

    try:
        cmd = f"""ffmpeg -i '{url}' \
                  -vf fps={fps} \
                  -hide_banner \
                  '{output}/{image_id}_frame%05d.png'"""

        print(cmd)

        proc = await asyncio.create_subprocess_shell(cmd)
        await proc.communicate()

    except Exception:
        return []

    frame_files = [f for f in os.listdir(
        output) if f.endswith('.png') and image_id in f]

    return frame_files


def get_vocab(txt_dir_path, filter_tags=None, top=None, splitter=", ", exts=None):
    # Also supports .vocab file:
    if os.path.isfile(txt_dir_path):
        with open(txt_dir_path, 'r') as f:
            data = f.read()
            return np.array(data.split(splitter))

    vocab = []
    occ = {}

    if exts is None:
        exts = ['.txt', '.text', '.tag']

    files = get_images(txt_dir_path, exts=exts)

    for file in files:
        with open(file, "r") as f:
            try:
                txt_data = f.read()
                txt_data = fix_tags(txt_data)
                tags = txt_data.split(', ')
                for tag in tags:

                    tag = tag.replace("\n", "")

                    if tag not in occ:
                        occ[tag] = 0

                    occ[tag] += 1

                    if filter_tags is not None:
                        for filter in filter_tags:
                            if filter not in tag:
                                continue
                            else:
                                vocab.append(tag)
                    else:
                        vocab.append(tag)
            except Exception:
                print(f"error processing {file}")

    vocab = list(set(vocab))

    if top is not None:
        new_vocab = []
        for k in sorted(occ, key=occ.get, reverse=True):
            if k in vocab:
                new_vocab.append(k)

            if len(new_vocab) >= top:
                break

        vocab = new_vocab

    return np.sort(np.array(vocab))


def tokenize(tags, vocab, pad=True, offset=0, size=None):

    if size is None:
        size = len(vocab)

    out = []

    for tag in tags:
        if tag in vocab:
            if isinstance(vocab, np.ndarray):
                out.append(np.where(vocab == tag)[0][0] + offset)
            else:
                out.append(vocab.index(tag) + offset)

    if len(out) < size:
        out += [0] * (len(vocab) - len(out))

    if len(out) > size:
        out = out[:size]

    return out


def decode(tags, vocab, padded=True, offset=0):

    decoded = []

    for tag in tags:
        if tag - offset > 0:
            decoded.append(vocab[tag - offset])

    return decoded


def fix_tags(txt_data):
    txt_data = txt_data.replace("tank_top", "tanktop")
    txt_data = txt_data.replace("bare_midriff", "midriff")
    txt_data = txt_data.replace("thong_panties", "thong")
    txt_data = txt_data.replace("age_rating_s, ", "")
    txt_data = txt_data.replace("age_rating_e, ", "")
    txt_data = txt_data.replace("age_rating_q, ", "")
    txt_data = txt_data.replace("meta_score_0, ", "")
    txt_data = txt_data.replace("original, ", "")
    txt_data = txt_data.replace("highres, ", "")
    txt_data = txt_data.replace("photo, ", "")
    txt_data = txt_data.replace("blond_hair", "blonde_hair")
    txt_data = txt_data.replace("erection_in_clothing",
                                "erection_under_clothing")
    txt_data = txt_data.replace("erection_in_clothes",
                                "erection_under_clothing")
    txt_data = txt_data.replace("erection_under_clothes",
                                "erection_under_clothing")
    txt_data = txt_data.replace("spread_legs", "legs_spread")
    txt_data = txt_data.replace("\n", "")
    txt_data = txt_data.replace("1girls", "1girl")
    txt_data = txt_data.replace("2girl,", "2girls,")
    txt_data = txt_data.replace("wood_floor", "wooden_floor")
    txt_data = txt_data.replace("pierced_navel", "navel_piercing")
    txt_data = txt_data.replace("1boys", "1boy")
    txt_data = txt_data.replace("off_the_shoulder", "off_shoulder")
    txt_data = txt_data.replace("carpet_floor", "carpet")
    txt_data = txt_data.replace("gray", "grey")
    txt_data = txt_data.replace("bare_foot", "barefoot")
    txt_data = txt_data.replace("tiles", "tile_floor")
    txt_data = txt_data.replace("closed_eyes", "eyes_closed")
    txt_data = txt_data.replace("nipple,", "nipples,")
    txt_data = txt_data.replace("twin_tails", "twintails")
    txt_data = txt_data.replace("pig_tails", "pigtails")
    txt_data = txt_data.replace("stone_walls", "stone_wall")
    txt_data = txt_data.replace("brick_walls", "brick_wall")
    txt_data = txt_data.replace("pink_walls", "pink_wall")
    txt_data = txt_data.replace("2girlsss", "2girls")
    txt_data = txt_data.replace("2girlss", "2girls")
    txt_data = txt_data.replace("nippless", "nipples")
    txt_data = txt_data.replace("erect_nipples_under_clothes", "nipple_bulge")
    txt_data = txt_data.replace("pov_male", "male_pov")
    txt_data = txt_data.replace("transparent_clothing", "see-through")

    if "erection" in txt_data and "erect_penis" not in txt_data:
        txt_data += ", erect_penis"

    if "cumshot" in txt_data and "ejaculation" not in txt_data:
        txt_data += ", ejaculation"

    return txt_data


def txt_to_onehot(vocab, txt, split=", ", trim=[" ", '\n']):

    if not isinstance(txt, str):
        raise ValueError("txt_to_onehot() expects a string blob as text input")

    onehot = np.zeros((len(vocab),), dtype=np.float)

    for t in txt.split(split):
        for tr in trim:
            t = t.replace(tr, "")

        match = np.where(vocab == t)[0]

        if len(match) == 0:
            continue

        match = match[0]

        onehot[match] = 1

    return onehot

    one_hot = {}

    # setup the hash
    for word in vocab:
        one_hot[word] = 0

    txt = txt.split(split)

    for word in txt:
        one_hot[word] = 1

    return np.array(list(one_hot.values()))


def txt_from_onehot(vocab, onehot, thresh=0.2, return_confidence=False):

    filtered = np.array(onehot) >= thresh

    txt = []
    conf = []

    for index in range(len(onehot)):
        if filtered[index]:
            txt.append(vocab[index])
            conf.append(onehot[index])

    if return_confidence:
        return txt, conf

    return txt


def onehot_to_image(onehot, img_shape, rgb=True):
    sqrt = round(math.pow(len(onehot), 0.5))

    sqr_y = sqrt * sqrt

    y = np.concatenate((onehot, np.zeros((sqr_y - len(onehot)))))

    y = np.reshape(y, (-1, sqrt))

    img = Image.fromarray(y * 127.5 + 1)
    img = img.convert("L")

    if rgb:
        img = img.convert("RGB")

    img = img.resize(img_shape, Image.BICUBIC)

    return np.asarray(img)


def image_combine(img1, img2, mask=[0, 0, 0], thresh=None):
    if thresh is None:
        img2_mask = np.any(img2 != mask, axis=-1)  # any non-black pixel
    else:
        # any pixel greater than thresh
        img2_mask = np.any(img2 >= thresh, axis=-1)

    final_img = img1.copy()

    final_img[img2_mask] = img2[img2_mask]

    return final_img


def get_images(path, exts=None, verify=False):

    images = []
    if exts is None:
        exts = [".png", ".jpg", ".webp", ".jpeg"]

    if isinstance(exts, str):
        exts = [exts]

    for root, dirs, files in os.walk(path, followlinks=True):
        for file in files:
            for ext in exts:
                if file.endswith(ext):
                    try:
                        fn = os.path.join(root, file)

                        if verify:
                            Image.open(fn)

                        images.append(fn)
                    except:
                        os.remove(fn)
                        continue

    return images


def load_image(img_file, shape=None, normalize=True, pixel_format='RGB'):
    # print(img_file)
    img = Image.open(img_file).convert(pixel_format)

    if shape is not None:
        img = img.resize(shape,
                         Image.BICUBIC)
    if normalize:
        img = np.asarray(img) / 127.5 - 1

    return img


class OtherCache:
    cache = {}


def get_txt_from_img_fn(img_fn, txts, mask=None, no_cache=False):

    img_bn = os.path.basename(img_fn)
    img_bn = os.path.splitext(img_bn)[0]

    if mask is not None:
        img_bn = img_bn.replace(mask, "")

    if not no_cache and img_bn in OtherCache.cache:
        return OtherCache.cache[img_bn]

    for txt in txts:
        bn = os.path.basename(txt)
        bn = os.path.splitext(bn)[0]

        OtherCache.cache[bn] = txt

        if mask is not None:
            bn = bn.replace(mask, "")

        # print(f"comparing {bn} ==? {img_bn}")

        if bn == img_bn:
            return txt


def test_onehot_to_image():

    onehot = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]

    import cv2

    image = onehot_to_image(onehot, (256, 256))

    cv2.imshow("muh onehot", image)
    cv2.waitKey(1000)


def test_vocab():

    vocab_dir = "."

    vocab = get_vocab(vocab_dir, top=1000)

    assert len(vocab) == 1000
