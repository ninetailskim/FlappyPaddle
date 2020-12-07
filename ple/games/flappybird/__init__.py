import os
import sys
import numpy as np

import pygame
from pygame.constants import K_w, K_ESCAPE, K_SPACE
from .. import base


class BirdPlayer(pygame.sprite.Sprite):

    def __init__(self,
                 SCREEN_WIDTH, SCREEN_HEIGHT, init_pos,
                 image_assets, rng, color="red", scale=1.0):

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.image_order = [0, 1, 2, 1]
        # done image stuff

        pygame.sprite.Sprite.__init__(self)

        self.image_assets = image_assets

        self.init(init_pos, color)

        self.height = self.image.get_height()
        self.scale = scale

        # all in terms of y
        self.vel = 0
        self.FLAP_POWER = 9 * self.scale
        self.MAX_DROP_SPEED = 10.0
        self.GRAVITY = 1.0 * self.scale

        self.rng = rng

        self._oscillateStartPos()  # makes the direction and position random
        self.rect.center = (self.pos_x, self.pos_y)  # could be done better

        self.lives = 1
        self.score = 0

    def init(self, init_pos, color):
        # set up the surface we draw the bird too
        self.flapped = True  # start off w/ a flap
        self.current_image = 0
        self.color = color
        self.image = self.image_assets[self.color][self.current_image]
        self.rect = self.image.get_rect()
        self.thrust_time = 0.0
        self.game_tick = 0
        self.pos_x = init_pos[0]
        self.pos_y = init_pos[1]
        self.lives = 1
        self.score = 0

    def _oscillateStartPos(self):
        offset = 8 * np.sin(self.rng.rand() * np.pi)
        self.pos_y += offset

    def flap(self):
        if self.pos_y > -2.0 * self.image.get_height():
            self.vel = 0.0
            self.flapped = True

    def update(self, dt):
        self.game_tick += 1

        # image cycle
        if (self.game_tick + 1) % 15 == 0:
            self.current_image += 1

            if self.current_image >= 2:
                self.current_image = 0

            # set the image to draw with.
            self.image = self.image_assets[self.color][self.current_image]
            self.rect = self.image.get_rect()

        if self.vel < self.MAX_DROP_SPEED and self.thrust_time == 0.0:
            self.vel += self.GRAVITY

        # the whole point is to spread this out over the same time it takes in
        # 30fps.
        if self.thrust_time + dt <= (1.0 / 30.0) and self.flapped:
            self.thrust_time += dt
            self.vel += -1.0 * self.FLAP_POWER
        else:
            self.thrust_time = 0.0
            self.flapped = False

        self.pos_y += self.vel
        self.rect.center = (self.pos_x, self.pos_y)

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)


class Pipe(pygame.sprite.Sprite):

    def __init__(self,
                 SCREEN_WIDTH, SCREEN_HEIGHT, gap_start, gap_size, image_assets, scale,
                 offset=0, color="green"):

        self.speed = 4.0 * scale
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.image_assets = image_assets
        # done image stuff

        self.width = self.image_assets["green"]["lower"].get_width()
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface((self.width, self.SCREEN_HEIGHT))
        self.image.set_colorkey((0, 0, 0))

        self.init(gap_start, gap_size, offset, color)

    def init(self, gap_start, gap_size, offset, color):
        self.image.fill((0, 0, 0))
        self.gap_start = gap_start
        self.x = self.SCREEN_WIDTH + self.width + offset

        self.lower_pipe = self.image_assets[color]["lower"]
        self.upper_pipe = self.image_assets[color]["upper"]

        top_bottom = gap_start - self.upper_pipe.get_height()
        bottom_top = gap_start + gap_size

        self.image.blit(self.upper_pipe, (0, top_bottom))
        self.image.blit(self.lower_pipe, (0, bottom_top))

        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.SCREEN_HEIGHT / 2)

    def update(self, dt):
        self.x -= self.speed
        self.rect.center = (self.x, self.SCREEN_HEIGHT / 2)


class Backdrop():

    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT,
                 image_background, image_base, scale):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.background_image = image_background
        self.base_image = image_base

        self.x = 0
        self.speed = 4.0 * scale
        self.max_move = self.base_image.get_width() - self.background_image.get_width()

    def update_draw_base(self, screen, dt):
        # the extra is on the right
        if self.x > -1 * self.max_move:
            self.x -= self.speed
        else:
            self.x = 0

        screen.blit(self.base_image, (self.x, self.SCREEN_HEIGHT * 0.79))

    def draw_background(self, screen):
        screen.blit(self.background_image, (0, 0))


class myUI():
    def __init__(self, width):
        pygame.font.init()
        self.width = width
        self.font = pygame.font.SysFont('microsoft Yahei',60)
        self.algoScore = 0
        self.humanScore = 0

    def draw_UI(self, algoScore, humanScore, screen):
        algoScoreUI = self.font.render(str(algoScore if algoScore > 0 else 0), False, (255, 0, 0))
        screen.blit(algoScoreUI, (0, 0))
        humanScoreUI = self.font.render(str(humanScore if humanScore > 0 else 0), False, (0, 0, 255))
        trec = humanScoreUI.get_rect()
        screen.blit(humanScoreUI, (self.width - trec.width, 0))


class FlappyBird(base.PyGameWrapper):
    """
    Used physics values from sourabhv's `clone`_.

    .. _clone: https://github.com/sourabhv/FlapPyBird


    Parameters
    ----------
    width : int (default: 288)
        Screen width. Consistent gameplay is not promised for different widths or heights, therefore the width and height should not be altered.

    height : inti (default: 512)
        Screen height.

    pipe_gap : int (default: 100)
        The gap in pixels left between the top and bottom pipes.

    """

    def __init__(self, width=288, height=512, pipe_gap=100):

        actions = {
             "_0": 0,
            "_1": 1,
        }

        fps = 30

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

        self.scale = 30.0 / fps

        self.allowed_fps = 30  # restrict the fps

        self.pipe_gap = pipe_gap
        self.pipe_color = "red"
        self.images = {}

        # so we can preload images
        pygame.display.set_mode((1, 1), pygame.NOFRAME)

        self._dir_ = os.path.dirname(os.path.abspath(__file__))
        self._asset_dir = os.path.join(self._dir_, "assets/")
        self._load_images()

        self.pipe_offsets = [0, self.width * 0.5, self.width]
        self.init_pos = (
            int(self.width * 0.2),
            int(self.height / 2)
        )

        self.pipe_min = int(self.pipe_gap / 4)
        self.pipe_max = int(self.height * 0.79 * 0.6 - self.pipe_gap / 2)

        self.backdrop = None
        self.player = None
        self.pipe_group = None

        self.myUI = myUI(self.width)

    def _load_images(self):
        # preload and convert all the images so its faster when we reset
        self.images["player"] = {}
        for c in ["red", "blue"]:
            image_assets = [
                os.path.join(self._asset_dir, "%spaddle-up.png" % c),
                os.path.join(self._asset_dir, "%spaddle-down.png" % c),
            ]

            self.images["player"][c] = [pygame.image.load(
                im).convert_alpha() for im in image_assets]

        self.images["background"] = {}
        for b in ["day", "night"]:
            path = os.path.join(self._asset_dir, "background-%s.png" % b)

            self.images["background"][b] = pygame.image.load(path).convert()

        self.images["pipes"] = {}
        for c in ["red", "green"]:
            path = os.path.join(self._asset_dir, "pipe-%s.png" % c)

            self.images["pipes"][c] = {}
            self.images["pipes"][c]["lower"] = pygame.image.load(
                path).convert_alpha()
            self.images["pipes"][c]["upper"] = pygame.transform.rotate(
                self.images["pipes"][c]["lower"], 180)

        path = os.path.join(self._asset_dir, "base.png")
        self.images["base"] = pygame.image.load(path).convert()

    def init(self):
        if self.backdrop is None:
            self.backdrop = Backdrop(
                self.width,
                self.height,
                self.images["background"]["day"],
                self.images["base"],
                self.scale
            )

        if self.player is None:
            self.player = [BirdPlayer(
                self.width,
                self.height,
                self.init_pos,
                self.images["player"],
                self.rng,
                color=cc,
                scale=self.scale
            ) for cc in ["red", "blue"]]

        if self.pipe_group is None:
            self.pipe_group = pygame.sprite.Group([
                self._generatePipes(offset=-75),
                self._generatePipes(offset=-75 + self.width / 2),
                self._generatePipes(offset=-75 + self.width * 1.5)
            ])

        color = self.rng.choice(["day"])
        self.backdrop.background_image = self.images["background"][color]

        # instead of recreating
        self.player[0].init(self.init_pos, "red")
        self.player[1].init(self.init_pos, "blue")

        self.pipe_color = self.rng.choice(["red", "green"])
        for i, p in enumerate(self.pipe_group):
            self._generatePipes(offset=self.pipe_offsets[i], pipe=p)

        self.score = 0.0
        self.lives = 1
        self.game_tick = 0

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        Returns
        -------

        dict
            * player y position.
            * players velocity.
            * next pipe distance to player
            * next pipe top y position
            * next pipe bottom y position
            * next next pipe distance to player
            * next next pipe top y position
            * next next pipe bottom y position


            See code for structure.

        """
        pipes0 = []
        for p in self.pipe_group:
            if p.x + p.width/2 > self.player[0].pos_x  :
                pipes0.append((p, p.x + p.width/2 - self.player[0].pos_x ))

        pipes0.sort(key=lambda p: p[1])

        next_pipe0 = pipes0[1][0]
        next_next_pipe0 = pipes0[0][0]

        if next_next_pipe0.x < next_pipe0.x:
            next_pipe0, next_next_pipe0 = next_next_pipe0, next_pipe0
        
        #####
        pipes1 = []
        for p in self.pipe_group:
            if p.x + p.width/2 > self.player[1].pos_x  :
                pipes1.append((p, p.x + p.width/2 - self.player[1].pos_x ))

        pipes1.sort(key=lambda p: p[1])

        next_pipe1 = pipes1[1][0]
        next_next_pipe1 = pipes1[0][0]

        if next_next_pipe1.x < next_pipe1.x:
            next_pipe1, next_next_pipe1 = next_next_pipe1, next_pipe1
        state = {
            "player0_y": self.player[0].pos_y,
            "player0_vel": self.player[0].vel,
            "next_pipe_dist_to_player0": next_pipe0.x + next_pipe0.width/2 - self.player[0].pos_x ,
            "player0_next_pipe_top_y": next_pipe0.gap_start,
            "player0_next_pipe_bottom_y": next_pipe0.gap_start + self.pipe_gap,
            "next_next_pipe_dist_to_player0": next_next_pipe0.x + next_next_pipe0.width/2 - self.player[0].pos_x ,
            "player0_next_next_pipe_top_y": next_next_pipe0.gap_start,
            "player0_next_next_pipe_bottom_y": next_next_pipe0.gap_start + self.pipe_gap,
            
            "player1_y": self.player[1].pos_y,
            "player1_vel": self.player[1].vel,
            "next_pipe_dist_to_player1": next_pipe1.x + next_pipe1.width/2 - self.player[1].pos_x ,
            "player1_next_pipe_top_y": next_pipe1.gap_start,
            "player1_next_pipe_bottom_y": next_pipe1.gap_start + self.pipe_gap,
            "next_next_pipe_dist_to_player1": next_next_pipe1.x + next_next_pipe1.width/2 - self.player[1].pos_x ,
            "player1_next_next_pipe_top_y": next_next_pipe1.gap_start,
            "player1_next_next_pipe_bottom_y": next_next_pipe1.gap_start + self.pipe_gap
        }

        return state

    def getScore(self):
        return max(self.player[0].score,self.player[1].score)

    def _generatePipes(self, offset=0, pipe=None):
        start_gap = self.rng.random_integers(
            self.pipe_min,
            self.pipe_max
        )

        if pipe is None:
            pipe = Pipe(
                self.width,
                self.height,
                start_gap,
                self.pipe_gap,
                self.images["pipes"],
                self.scale,
                color=self.pipe_color,
                offset=offset
            )

            return pipe
        else:
            pipe.init(start_gap, self.pipe_gap, offset, self.pipe_color)

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == 1:
                    self.player[0].flap()
                if key == K_w or key == K_SPACE:
                    self.player[1].flap()
                if key == K_ESCAPE:
                    self._setup()
                    self.init()
                    self.reset()


    def game_over(self):
        return self.player[0].lives + self.player[1].lives <= 0

    def step(self, dt):
        self.game_tick += 1
        dt = dt / 1000.0

        self.player[0].score += self.rewards["tick"]
        self.player[1].score += self.rewards["tick"]

        # handle player movement
        self._handle_player_events()

        for p in self.pipe_group:
            if self.player[0].lives > 0:
                hit = pygame.sprite.spritecollide(
                    self.player[0], self.pipe_group, False)

                is_in_pipe = (p.x - p.width/2 - 20) <= self.player[0].pos_x < (p.x + p.width/2)
                for h in hit:  # do check to see if its within the gap.
                    top_pipe_check = (
                        (self.player[0].pos_y - self.player[0].height/2 + 12) <= h.gap_start) and is_in_pipe
                    bot_pipe_check = (
                        (self.player[0].pos_y +
                        self.player[0].height) > h.gap_start +
                        self.pipe_gap) and is_in_pipe

                    if top_pipe_check:
                        self.player[0].lives -= 1

                    if bot_pipe_check:
                        self.player[0].lives -= 1

                # is it past the player?
                if (p.x - p.width / 2) <= self.player[0].pos_x < (p.x - p.width / 2 + 4):
                    self.player[0].score += self.rewards["positive"]

                # is out out of the screen?
                if p.x < -p.width:
                    self._generatePipes(offset=self.width * 0.2, pipe=p)

            ##############
            if self.player[1].lives > 0:
                hit = pygame.sprite.spritecollide(
                    self.player[1], self.pipe_group, False)

                is_in_pipe = (p.x - p.width/2 - 20) <= self.player[1].pos_x < (p.x + p.width/2)
                for h in hit:  # do check to see if its within the gap.
                    top_pipe_check = (
                        (self.player[1].pos_y - self.player[1].height/2 + 12) <= h.gap_start) and is_in_pipe
                    bot_pipe_check = (
                        (self.player[1].pos_y +
                        self.player[1].height) > h.gap_start +
                        self.pipe_gap) and is_in_pipe

                    if top_pipe_check:
                        self.player[1].lives -= 1

                    if bot_pipe_check:
                        self.player[1].lives -= 1

                # is it past the player?
                if (p.x - p.width / 2) <= self.player[1].pos_x < (p.x - p.width / 2 + 4):
                    self.player[1].score += self.rewards["positive"]

                # is out out of the screen?
                if p.x < -p.width:
                    self._generatePipes(offset=self.width * 0.2, pipe=p)

        # fell on the ground
        drawback = True
        pgu = True
        if self.player[0].lives > 0:
            if self.player[0].pos_y >= 0.79 * self.height - self.player[0].height:
                self.player[0].lives -= 1

            # went above the screen
            if self.player[0].pos_y <= 0:
                self.player[0].lives -= 1

            self.player[0].update(dt)
            if pgu:
                self.pipe_group.update(dt)
                pgu = False

            if self.player[0].lives <= 0:
                self.player[0].score += self.rewards["loss"]
            if drawback:
                self.backdrop.draw_background(self.screen)
                self.pipe_group.draw(self.screen)
                self.backdrop.update_draw_base(self.screen, dt)
                drawback = False
            self.player[0].draw(self.screen)

        #############
        # fell on the ground
        if self.player[1].lives > 0:
            if self.player[1].pos_y >= 0.79 * self.height - self.player[1].height:
                self.player[1].lives -= 1

            # went above the screen
            if self.player[1].pos_y <= 0:
                self.player[1].lives -= 1

            self.player[1].update(dt)
            if pgu:
                self.pipe_group.update(dt)
                pgu = False

            if self.player[1].lives <= 0:
                self.player[1].score += self.rewards["loss"]

            if drawback:
                self.backdrop.draw_background(self.screen)
                self.pipe_group.draw(self.screen)
                self.backdrop.update_draw_base(self.screen, dt)
                drawback = False
            self.player[1].draw(self.screen)
        
        self.myUI.draw_UI(int(self.player[0].score), int(self.player[1].score), self.screen)
