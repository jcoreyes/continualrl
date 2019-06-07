#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid

def main():
	parser = OptionParser()
	parser.add_option(
		"-e",
		"--env-name",
		dest="env_name",
		help="gym environment to load",
		default='MiniGrid-MultiRoom-N6-v0'
	)
	(options, args) = parser.parse_args()

	# Load the gym environment
	env = gym.make(options.env_name)

	def resetEnv():
		env.reset()
		if hasattr(env, 'mission'):
			print('Mission: %s' % env.mission)

	resetEnv()

	# Create a window to render into
	renderer = env.render('human')

	def keyDownCb(keyName):
		if keyName == 'BACKSPACE':
			resetEnv()
			return

		if keyName == 'ESCAPE':
			sys.exit(0)

		action = 0

		if keyName == 'LEFT':
			action = env.actions.west
		elif keyName == 'RIGHT':
			action = env.actions.east
		elif keyName == 'UP':
			action = env.actions.north
		elif keyName == 'DOWN':
			action = env.actions.south

		elif keyName == 'SPACE':
			action = env.actions.mine
		elif keyName == 'PAGE_UP':
			action = env.actions.eat
		elif keyName == '0':
			action = env.actions.place0
		elif keyName == '1':
			action = env.actions.place1
		elif keyName == '2':
			action = env.actions.place2
		elif keyName == '3':
			action = env.actions.place3
		elif keyName == '4':
			action = env.actions.place4
		elif keyName == 'RETURN':
			action = env.actions.done

		else:
			print("unknown key %s" % keyName)
			return

		obs, reward, done, info = env.step(action)

		print('step=%s, reward=%.2f, health=%d' % (env.step_count, reward, env.health))

		if done:
			print('done!')
			resetEnv()

	renderer.window.setKeyDownCb(keyDownCb)

	while True:
		env.render('human')
		time.sleep(0.01)

		# If the window was closed
		if renderer.window == None:
			break

if __name__ == "__main__":
	main()
