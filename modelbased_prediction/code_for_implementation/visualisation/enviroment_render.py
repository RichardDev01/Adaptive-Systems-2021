"""This file contains code to render a maze."""

from pathlib import Path

from PIL import Image, ImageDraw

textures_path = Path(__file__) / '..' / 'textures'


def render_background(environment):
    """
    Render background used for the simluation.

    :param environment: The environment given to render
    :return: Background as pillow image
    """
    # Loading textures in
    path = Image.open(textures_path / "dirt.png")
    exit_s = Image.open(textures_path / "objective_marker.png")

    # Get width and height of the images
    tile_size_width, tile_size_height = path.size

    # Variable declaration
    maze = environment.maze  # np.array

    # Calculate canvas size of the maze
    width = maze.shape[0] * tile_size_width
    height = maze.shape[1] * tile_size_height

    # Get maze grind size
    maze_width, maze_height = maze.shape

    # Create canvas
    background = Image.new(mode="RGB", size=(width, height))

    # Loop through values from the maze and determine layout
    for height_row, width_values in enumerate(maze):
        for index, _ in enumerate(width_values):
            background.paste(path, (index * tile_size_width, height_row * tile_size_height), path)

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
    """Render the things that needs to be updated every step."""
    # Copy from background
    copy_background = environment.rendered_background.copy()
    # Declare with images used for the agent
    agent_icon = Image.open(textures_path / "agent.png")

    # Calculate canvas size of the maze
    tile_width = environment.rendered_background.width // environment.maze.shape[0]
    # tile_height = environment.rendered_background.height // environment.maze.shape[1]

    # Get width and height of the images
    tile_size_width, tile_size_height = agent_icon.size

    # # Loop through values from the maze and determine layout
    # for height_row, width_values in enumerate(environment.occupied_map):
    #     for index, occupation_value in enumerate(width_values):
    #         if occupation_value > 0:
    #             copy_background.paste(agent_icon, (index * tile_size_width, height_row * tile_size_height), agent_icon)

    # Draw location of the agent in the maze
    copy_background.paste(agent_icon, (
        environment.agent_location[0] * tile_size_width, environment.agent_location[1] * tile_size_height), agent_icon)

    # Draw current step and reword on the screen
    ImageDraw.Draw(copy_background).text((5, 0), f"Time: {environment.sim_step}\nReward: {environment.total_reward}")

    # Translate action to text and write to screen
    action_to_string_dict = {0: 'up',
                             1: 'right',
                             2: 'down',
                             3: 'left',
                             4: 'stay'}

    ImageDraw.Draw(copy_background).text((environment.rendered_background.width - 2 * tile_width, 0),
                                         f"Last action: {action_to_string_dict[environment.last_action_agent]}")
    return copy_background
