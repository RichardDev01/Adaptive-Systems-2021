"""This file contains code to render a maze."""

from pathlib import Path

from PIL import Image, ImageDraw

textures_path = Path(__file__) / '..' / 'textures'


def render_background(environment):
    """
    Render background used for the simulation.

    :param environment: The environment given to render
    :return: Background as pillow image
    """
    # Loading textures in
    path = Image.open(textures_path / "grass.png")
    water = Image.open(textures_path / "water.png")
    highlight = Image.open(textures_path / "highlighter.png")
    exit_s = Image.open(textures_path / "objective_marker.png")

    # Get width and height of the images
    tile_size_width, tile_size_height = path.size

    # Variable declaration
    maze = environment.reward_map  # np.array

    # Calculate canvas size of the maze
    width = maze.shape[0] * tile_size_width
    height = maze.shape[1] * tile_size_height

    # Get maze grind size
    maze_width, maze_height = maze.shape

    # Create canvas
    background = Image.new(mode="RGB", size=(width, height))

    # Loop through values from the maze and determine layout
    for height_row, width_values in enumerate(maze):
        for index, reward_value in enumerate(width_values):
            if reward_value == -10:
                background.paste(water, (index * tile_size_width, height_row * tile_size_height), water)
            else:
                background.paste(path, (index * tile_size_width, height_row * tile_size_height), path)
            background.paste(highlight, (index * tile_size_width, height_row * tile_size_height), highlight)
    # Loop through values from the maze and determine reward table
    for height_row, width_values in enumerate(environment.reward_map):
        for index, rewards in enumerate(width_values):
            ImageDraw.Draw(background).text((index * tile_size_width + tile_size_width / 4,
                                             height_row * tile_size_height + tile_size_height / 2),
                                            f"R ={rewards}")

    # Render exit of the maze with different marker
    for exit_maze in environment.end_coord:
        background.paste(exit_s,
                         (exit_maze[0] * tile_size_width + tile_size_width // 8,
                          exit_maze[1] * tile_size_height + tile_size_height // 8),
                         exit_s)

    return background


def render_in_step(environment):
    """
    Render the things that needs to be updated every step.

    :param environment: The environment given to render
    :return: Background with sprites in step rendered as pillow image
    """
    # Copy from background
    copy_background = environment.rendered_background.copy()
    # Declare with images used for the agent
    agent_icon = Image.open(textures_path / "agent.png")

    # Calculate canvas size of the maze
    tile_width = environment.rendered_background.width // environment.maze.shape[0]

    # Get width and height of the images
    tile_size_width, tile_size_height = agent_icon.size

    # Draw current step and reword on the screen
    ImageDraw.Draw(copy_background).text((5, 0), f"Time: {environment.sim_step}\nReward: {environment.total_reward}")

    # Translate action to text and write to screen
    action_to_string_dict = {0: 'up',
                             1: 'right',
                             2: 'down',
                             3: 'left',
                             4: 'stay',
                             None: 'None'}

    ImageDraw.Draw(copy_background).text((environment.rendered_background.width - 2 * tile_width, 0),
                                         f"Last action: {action_to_string_dict[environment.last_action_agent]}")

    # Draw actions according to the value matrix
    if environment.agent.policy.visual_matrix is not None:
        # Loop through values from the maze and determine reward table
        for height_row, width_values in enumerate(environment.agent.policy.visual_matrix):
            for index, action in enumerate(width_values):
                ImageDraw.Draw(copy_background).text((index * tile_size_width + tile_size_width / 4,
                                                      height_row * tile_size_height + tile_size_height / 3),
                                                     f"A ={action}")

        # Draw location of the agent in the maze
    copy_background.paste(agent_icon,
                          (environment.agent_location[1] * tile_size_width,
                           environment.agent_location[0] * tile_size_height),
                          agent_icon)

    return copy_background
