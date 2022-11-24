"""Script to edit videos."""

import moviepy.editor as mpy


def video_editor(title, runs, videos, path):
    """Compare multiple video runs.

    It reads different videos and creates a side-by-side comparison to
    truly how the agents are executing the different policies.

    Parameters
    ----------
    title : str
        Title of the video, as well as of the file to be saved.
    runs : list
        each list element should contain a string, e.g., action,
        reconstruction, etc.
    videos : dict
        This dictionary has all the paths to read the videos.
    path : string
        folder where the video will be saved

    Examples
    --------
    title = 'VAE models'
    runs = ['Action', 'Recon']

    models = ['l3_len4_z12_e320', 'l2_len8_z12_e400', 'l1_len64_z12_e400']
    m_names = ['3 Levels', '2 Levels', '1 Level']

    kinds = [['best_action', 'best_recon'], ['worst_action', 'worst_recon']]
    k_names = ['Best', 'Worst']

    videos = {}

    for model, m_name in zip(models, m_names):
        for kind, k_name in zip(kinds, k_names):
            paths = []
            for k in kind:
                paths.append(f'{model}/{k}')
            videos[f'{m_name} {k_name}'] = paths


    video_editor(title, runs, videos, 'cool1')
    """
    p = 'Videos'
    name = 'rl-video-episode-0.mp4'
    clips = []
    size = len(list(videos.values())[0])
    size = (1000, 500) if size == 2 else (1500, 500)
    Title = create_title(title, size, 3)
    clips.append(Title)

    for subtitle, vids in videos.items():
        sub = create_title(subtitle, size, 1.5)
        clips.append(sub)
        aux_vid = []
        for run, vid in zip(runs, vids):
            c_vid = mpy.VideoFileClip(f'{p}/{vid}/{name}')
            txt = create_text(run, c_vid.duration)
            c_vid = mpy.CompositeVideoClip([c_vid, txt], size=(500, 500))
            aux_vid.append(c_vid)

        combined = mpy.clips_array([aux_vid])
        clips.append(combined)

    final_clip = mpy.concatenate_videoclips(clips)
    final_clip.write_videofile(f'{p}/{path}/{title}.mp4')


def create_title(title, size, duration):
    """Create title for video.

    It creates a short video with black background and a title.

    Parameters
    ----------
    title : str
        The title of the video
    size : tuple
        The frame of the video, e.g., (1000, 500).
    duration : float
        How long the title video will be displayed.
    """
    txt = mpy.TextClip(title, fontsize=80, color='White')
    txt = txt.set_position(('center', 'center'), relative=True)
    txt = txt.set_duration(duration)

    return mpy.CompositeVideoClip([txt], size=size)


def create_text(run, duration):
    """Create text to overlay.

    Parameters
    ----------
    run : str
        String for the text.
    duration : float
        How long the text will be displayed. By default, it is the
        same as the duration of the video.
    """
    txt = mpy.TextClip(run, fontsize=60, color='Black')
    txt = txt.set_position(('center', 'top'), relative=True)
    txt = txt.set_duration(duration)
    return txt


title = 'VAE models 16 skill'
runs = ['Action', 'Recon']

models = ['l3_len4_z12_e320', 'l2_len8_z12_e400', 'l1_len64_z12_e400']
m_names = ['3 Levels', '2 Levels', '1 Level']

kinds = [['best_action', 'best_recon'], ['worst_action', 'worst_recon']]
k_names = ['Best', 'Worst']

videos = {}

for model, m_name in zip(models, m_names):
    for kind, k_name in zip(kinds, k_names):
        paths = []
        for k in kind:
            paths.append(f'{model}/{k}')
        videos[f'{m_name} {k_name}'] = paths


video_editor(title, runs, videos, 'Comparisons')
