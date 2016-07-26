"""
Cross-origin Server
===================
This is a minimal server for interfacing OpenAI Gym with MDPvis.
Experiment with MDPvis at MDPvis.github.io.

:author: Sean McGregor.
"""
from flask import Flask, jsonify, request, send_file, make_response
from flask.ext.cors import cross_origin
import subprocess
import argparse
import numpy
from PIL import Image
from StringIO import StringIO
import gym

print "Starting Flask Server at http://localhost:8938"
app = Flask('openaigym', static_folder='.', static_url_path='')
parser = argparse.ArgumentParser(description='Start the OpenAI gym server.')
parser.add_argument('domain', metavar='D', type=str,
                    help='the domain to synthesize trajectories for',
                    default='CartPole-v0')
parser.add_argument('outdir', metavar='O', type=str,
                    help='the directory used to temporarily store videos rendered within MDPvis',
                    default='/tmp/random-agent-results')
parser.add_argument('perception', metavar='P', type=bool,
                    help='flag indicates whether the domain has raw perception state, such as an Atari domain',
                    default=False)
parser.add_argument('names', metavar='N', type=str, nargs='*',
                    help='the names of the dimensions of the state')
args = vars(parser.parse_args())
domain_name = args["domain"]
dimension_names = args["names"]
perception = args["perception"]
output_directory = args["outdir"]

print "Starting {} with variable names {}".format(domain_name, dimension_names)
print "You can start other domains with the following commands:\n"
print "python flask_server.py CartPole-v0 /tmp/random-agent-results True x x_dot theta theta_dot"
print "python flask_server.py Pendulum-v0 /tmp/random-agent-results True cos(theta) sin(theta) thetadot"
print "python flask_server.py MountainCar-v0 /tmp/random-agent-results True position velocity"
print "python flask_server.py Acrobot-v0 /tmp/random-agent-results True angle1 angle2 speed1 speed2"
print "python flask_server.py MsPacman-v0 /tmp/random-agent-results True"
print "\n\nOther domains are possible if you change the domain and dimension names appropriately."


class RandomAgent(object):
    """
    Placeholder randomized policy function. Implement or integrate your own policy.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def similarity(state1, state2):
    """
    Recursively find the similarity of two observations as the sum of L1 distance across all pairs of pixels.
    This is currently implemented for the Arcade Learning Environment domains.
    :param state1: A number, a list of numbers, or a list of lists
    :param state2: A number, a list of numbers, or a list of lists
    :return: an integer sum of distances.
    """
    total = 0
    if type(state1) is list or type(state1) is numpy.ndarray:
        for idx, s1 in enumerate(state1):
            total += similarity(s1, state2[idx])
    else:
        return abs(state1 - state2)
    return total


def _simulate_trajectory(seed, parameters, generate_video=False):
    """
    Generate a single trajectory for a specific seed value to the specified horizon.
    :param seed: The seed to start the monitor and policy with.
    :param parameters: The parameters of the HTTP request as sent by MDPvis.
      This must include the argument "Horizon," which is expected to be a non-zero integer passed as a string.
    :param generate_video: Whether the video should be written to disk.
    :return: An array of dictionaries. Each time step has its own dictionary keyed by the dimensions given
      when starting the server. Specify the dimensions with the `N` argument on the command line. The additional
      dimension of "image row" will be added to the first time step of every trajectory. The image row stores the
      seed and horizon parameters for regenerating videos of the trajectory at a later point.
    """
    env = gym.make(domain_name)
    max_steps = int(parameters["Horizon"])
    env.monitor.start(output_directory,
                      force=True,
                      seed=seed,
                      video_callable=lambda count: generate_video)
    agent = RandomAgent(env.action_space)
    agent.action_space.seed_prng(seed)

    trajectory = []
    reward = 0
    done = False
    ob = env.reset()
    for i in range(max_steps):
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        current_time_step = {}
        if perception:
            if i is 0:
                initial_frame = ob
            sim = similarity(initial_frame, ob)
            current_time_step["initial.similarity.png"] = sim
        else:
            for idx, x in enumerate(ob):
                current_time_step[dimension_names[idx]] = x
        current_time_step["reward"] = reward
        trajectory.append(current_time_step)
        if done:
            break
    trajectory[0]["image row"] = [str(seed) + "." + str(max_steps) + ".mp4"]
    if generate_video:
        video_file_path = env.monitor.video_recorder.path
    else:
        video_file_path = None
    env.monitor.close()  # Dump result info to disk

    return trajectory, video_file_path

@app.route("/", methods=['GET'])
@cross_origin()
def site_root():
    '''
        Return instructions on how to request data from the other endpoints.
    '''
    return '''
        <h1>Hello World!</h1>
        <p style='font-size: 150%;'>Your server is running and ready for
        integration.</p>
        <p  style='font-size: 150%;'>To test the other endpoints, visit
          <a href="/initialize">/initialize</a>,
          <a href="/trajectories">/trajectories</a>,
          <a href="/optimize">/optimize</a>, or
          <a href="/state">/state</a>
        '''

@app.route("/initialize", methods=['GET'])
@cross_origin(allow_headers=['Content-Type'])
def cross_origin_initialize():
    '''
        Asks the domain for the parameters to seed the visualization.
    '''
    # The initialization object for MDPvis
    mdpvis_initialization_object = {

        # The control panels that appear at the top of the screen
        "parameter_collections": [
            {
                "panel_title": "Sampling Effort",
                "panel_icon": "glyphicon-retweet",
                "panel_description": "Define how many trajectories you want to generate, and to what time horizon.",
                "quantitative": [  # Real valued parameters
                                   {
                                       "name": "Sample Count",
                                       "description": "Specify how many trajectories to generate",
                                       "current_value": 10,
                                       "max": 1000,
                                       "min": 1,
                                       "step": 10,
                                       "units": "#"
                                   },
                                   {
                                       "name": "Horizon",
                                       "description": "The time step at which simulation terminates",
                                       "current_value": 10,
                                       "max": 10000,
                                       "min": 1,
                                       "step": 10,
                                       "units": "Time Steps"
                                   },
                                   {
                                       "name": "Seed",
                                       "description": "The random seed used for simulations",
                                       "current_value": 0,
                                       "max": 100000,
                                       "min": 1,
                                       "step": 1,
                                       "units": "NA"
                                   }
                ]
            }
        ]
    }
    return jsonify(mdpvis_initialization_object)

@app.route("/trajectories", methods=['GET'])
@cross_origin(allow_headers=['Content-Type'])
def cross_origin_trajectories():
    '''
        Asks the domain for the trajectories generated by the
        requested parameters.
    '''
    episode_count = int(request.args["Sample Count"])
    trajectories = []
    for i in range(episode_count):
        trajectory, _ = _simulate_trajectory(i, request.args, generate_video=False)
        trajectories.append(trajectory)
    json_obj = {"trajectories": trajectories}
    resp = jsonify(json_obj)
    return resp

@app.route("/optimize", methods=['POST','GET'])
@cross_origin(allow_headers=['Content-Type'])
def cross_origin_optimize():
    '''
        Asks for a newly optimized policy. You should integrate whatever optimization library you are developing
        here. This is a filler function since OpenAI Gym is a collection of domains without optimization algorithms.
    '''
    raise NotImplementedError
    count = int(request.args["Sample Count"])
    horizon = int(request.args["Horizon"])
    runs_limit = int(request.args["Number of Runs Limit"]) # Control how much effort is expended in optimization
    subprocess.call(
        ['./CALL_OPTIMIZATION_ALGORITHM'],
        shell=True)
    f = open("OPTIMIZATION.results", "r")
    resp = jsonify(f)
    f.close()
    return resp

@app.route("/state", methods=['POST','GET'])
@cross_origin(allow_headers=['Content-Type'])
def cross_origin_state():
    '''
        Generate the video associated with a trajectory and return it to MDPvis.
    '''
    image_details = request.args["image"].split(".")
    seed = int(image_details[0])
    horizon = int(image_details[1])
    _, video_file_path = _simulate_trajectory(seed, {"Horizon": horizon}, generate_video=True)
    print "sending %s" % video_file_path
    return send_file(video_file_path, mimetype='video/mp4')

@app.route("/initial.similarity.png", methods=['POST','GET'])
@cross_origin(allow_headers=['Content-Type'])
def cross_origin_similarity():
    '''
        Get the image associated with the similarity measure for the current state.
        This is currently only the first frame of the game.
    '''
    seed = 0
    env = gym.make(domain_name)
    env.monitor.start(output_directory,
                      force=True,
                      seed=seed,
                      video_callable=lambda count: False)
    env.reset()
    env.render()
    img_array = env._get_image()
    env.monitor.close()  # Dump result info to disk
    img = Image.fromarray(img_array, 'RGB')
    image_file = StringIO()  # writable object
    img.save(image_file, format='PNG')
    image_file.seek(0)

    response = make_response(send_file(image_file, mimetype='image/png'))
    response.headers["Pragma-directive"] = "no-cache"
    response.headers["Cache-directive"] = "no-cache"
    response.headers["Cache-control"] = "no-cache"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Binds the server to port 8938 and listens to all IP addresses.
if __name__ == "__main__":
    print("Running App...")
    app.run(host='0.0.0.0', port=8938, debug=True, use_reloader=False, threaded=True)
