import sys
import os
import tempfile

import matplotlib.pyplot as plt

__file__ = '/home/seongwoo/workspace/SLACK-util-test/model_noti/slack_handler/'
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/home/seongwoo/workspace/SLACK-util-test/model_noti'))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from slack_handler.slack_channel_noti import post_message
from slack_handler import config
from slack_handler.slack_channel_noti import post_file


def send_plot(train_loss_list, val_loss_list):
    with tempfile.TemporaryDirectory() as tempDir:
        if os.path.exists(tempDir):
            plt.plot(train_loss_list)
            plt.plot(val_loss_list)
            plt.savefig(f'{tempDir}/val_plot.png')
            with open(f'{tempDir}/val_plot.png', 'rb') as f:
                content = f.read()
                post_file(config.user_token, config.channel_id, f.name, content)

def _post_message(post_msg):
    return post_message(config.token, config.channel_id, post_msg)