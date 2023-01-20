import os
import shutil

from GelbooruViewer import GelbooruViewer
from urllib.request import urlretrieve, build_opener, install_opener
from PIL import Image

def is_video(url):
    return (url.endswith(".webm") or
            url.endswith(".gif"))

def split_animation(url, image_id, output, fps):

    import subprocess

    try:
        subprocess.run("ffmpeg -i '{}' -vf fps={} '{}/{}_frame%05d.png' -hide_banner".format(
            url, fps, output, image_id), shell=True, check=True)
    except Exception:
        return []

    frame_files = [f for f in os.listdir(
        output) if f.endswith('.png') and image_id in f]

    return frame_files


def get_artists_from_path(path):
    dirs = next(os.walk(path, followlinks=True))[1]

    return dirs


def get_images(txt_out, img_out, tags, num=1000, start_id=0, no_animated=False, fps=5,
               danbooru=False):

    opener = build_opener()
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    install_opener(opener)

    if txt_out is None:
        txt_out = os.path.join(img_out, "tags")

    os.makedirs(txt_out, exist_ok=True)
    os.makedirs(img_out, exist_ok=True)

    count = 0
    posts = []

    if not danbooru:
        viewer = GelbooruViewer()
        pictures = viewer.get_all(tags=tags, num=num, pid=start_id, limit=100)

        if pictures is None:
            print(f"no images for tags: {tags}")
            return

        for pic in pictures:

            img_id = pic.picture_id
            tags = ", ".join(pic.tags)
            url = pic.file_url
            url = url.replace("https:https", "https")
            posts.append((url, pic.tags, img_id))

    else:
        from pybooru import Danbooru

        client = Danbooru('danbooru')
        pictures = client.post_list(tags=tags, limit=num)

        if pictures is None:
            print(f"no images for tags: {tags}")
            return

        for pic in pictures:
            if "large_file_url" not in pic:
                continue
            img_id = pic["id"]
            tags = pic["tag_string"].split(" ")
            posts.append((pic['large_file_url'], 
                          ", ".join(tags), 
                          img_id))


    for url, tags, img_id in posts:
        print(url)
        fn, ext = os.path.splitext(url)
        img_out_tmp = os.path.join(img_out, "{}{}".format(img_id, ext))

        txt_out_tmp = os.path.join(txt_out, "{}.txt".format(img_id))

        if os.path.exists(txt_out_tmp):
            print('skipping {} because it already exists'.format(img_id))
            continue

        count += 1

        with open(txt_out_tmp, 'w') as f:
            if isinstance(tags, list):
                tags = ", ".join(tags)
            f.write(tags)

        if not no_animated and ("animated" in tags or is_video(url)):
            frames = split_animation(url, img_id, img_out, fps)

            count += len(frames)

            for frame in frames:
                bn = os.path.basename(frame)
                bn = bn.replace("png", "txt")
                shutil.copy2(txt_out_tmp, os.path.join(txt_out, bn))

        elif not is_video(url):
            try:
                urlretrieve(url, img_out_tmp)
                # Verify the image is actually an image...
                try:
                    Image.open(img_out_tmp)
                except:
                    print(f"Image {img_out_tmp} probably didn't download correctly. Removing...")
                    os.remove(img_out_tmp)
                    os.remove(txt_out_tmp)
            except Exception:
                pass

    return count


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--txt', dest='txt_out', help='txt output directory')
    parser.add_argument('--img', dest='img_out', help='image output directory')
    parser.add_argument('--num', dest="num", type=int,
                        help='number of images to fetch',
                        default=1000)

    parser.add_argument('--start_id', dest='start_id', type=int, default=0)
    parser.add_argument('--url', default=None)
    parser.add_argument('--tags', nargs="+")
    parser.add_argument('--no_animated', action="store_true")
    parser.add_argument('--fps', type=int, default=5,
                        help="number of frames for animation")

    parser.add_argument('--update_artists', action='store_true')
    parser.add_argument('--danbooru', action='store_true')

    args = parser.parse_args()

    tags = args.tags

    jobs = [(args.txt_out, args.img_out, tags)]

    GelbooruViewer.API_URL = args.url
    if args.url == "realbooru":
        GelbooruViewer.API_URL = "https://realbooru.com/index.php?page=dapi&s=post&q=index"
    elif args.url is None:
        GelbooruViewer.API_URL = "https://rule34.xxx/index.php?page=dapi&s=post&q=index"

    if args.update_artists:
        artists = get_artists_from_path(args.img_out)

        print(artists)

        jobs = []

        for artist in artists:
            txt_out = os.path.join(args.img_out, artist, "txts")
            img_out = os.path.join(args.img_out, artist, "imgs")

            job = (txt_out, img_out, [artist])
            print(job)

            jobs.append(job)

    count = 0

    for job in jobs:

        txt_out, img_out, tags = job

        added = get_images(txt_out, img_out, tags, args.num, args.start_id,
                           args.no_animated, args.fps, args.danbooru)

        if added is not None:
            count += added

    print(f"added {count} images")


if __name__ == '__main__':
    main()
