import sys
import random
import pygame
import chicken_run_utils
from pygame.locals import *
from itertools import cycle

FPS = 30
SCREENWIDTH  =  512
SCREENHEIGHT =  288

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Chicken Run')

IMAGES, SOUNDS, HITMASKS = chicken_run_utils.load()
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
BARRIER_WIDTH = IMAGES['barrier'][0].get_width()
BARRIER_HEIGHT = (
    IMAGES['barrier'][0].get_height(),
    IMAGES['barrier'][1].get_height(),
    IMAGES['barrier'][2].get_height(),
)
BACKGROUND_WIDTH = IMAGES['background'].get_width()
BASE_HEIGHT = IMAGES['base'].get_height()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int(BASEY - PLAYER_HEIGHT)

        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newBar1 = getRandomBarrier()
        newBar2 = getRandomBarrier()

        gapXs = [150, 250, 350]  # change here
        index = random.randint(0, len(gapXs) - 1)
        X = SCREENWIDTH + gapXs[index]
        self.uppers = [
            {'x': SCREENWIDTH, 'y': newBar1[0]['y'], 'type': newBar1[0]['type']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newBar2[0]['y'], 'type': newBar2[0]['type']},
        ]
        self.lowers = [
            {'x': SCREENWIDTH, 'y': newBar1[1]['y'], 'type': newBar1[1]['type']},
            {'x': X, 'y': newBar2[1]['y'], 'type': newBar2[1]['type']},
        ]


        # player velocity, max velocity, downward accleration, accleration when jump
        self.velX = -8
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  100   # max vel along Y, max descend speed
        self.playerMinVelY =  -80   # min vel along Y, max ascend speed
        self.playerAccY    =   2   # players downward accleration
        self.playerJumpAcc =  -22   # players upward accleration when jump
        self.playerJumped = False # True when player jumps



    def frame_step(self, input_actions):
        # pygame.event.pump()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                input_actions[1] = 1
                input_actions[0] = 0
            else:
                input_actions[1] = 0
                input_actions[0] = 1

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: jump
        if input_actions[1] == 1 and self.playery >= int(BASEY - PLAYER_HEIGHT): # jump when it's on the ground
            self.playerVelY = self.playerJumpAcc
            self.playerJumped = True
            #SOUNDS['wing'].play()

        # check for score
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for bar in self.lowers:
            barMidPos = bar['x'] + BARRIER_WIDTH / 2
            if barMidPos <= playerMidPos < barMidPos - self.velX:
                self.score += 1
                reward = 1
                # SOUNDS['point'].play()

        # # playerIndex basex change
        # if (self.loopIter + 1) % 3 == 0:
        #     self.playerIndex = next(PLAYER_INDEX_GEN)
        # self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerJumped:
            self.playerVelY += self.playerAccY
        if self.playerJumped:
            self.playerJumped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT) # on the ground
        if self.playery < 0:
            self.playery = 0
        if self.playery == 0:
            self.playerFlapped = False

        # move barriers to left
        for u, l in zip(self.uppers, self.lowers):
            u['x'] += self.velX
            l['x'] += self.velX

        # add new bar when the first one is about to touch the left of screen
        if 0 < self.uppers[0]['x'] < 10:
            newBar = getRandomBarrier()
            self.uppers.append(newBar[0])
            self.lowers.append(newBar[1])

        # remove first bar if it's out of the screen
        if self.uppers[0]['x'] < -BARRIER_WIDTH:
            self.uppers.pop(0)
            self.lowers.pop(0)

        # check if crash here
        isCrash = checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.uppers, self.lowers)
        if isCrash:
            #SOUNDS['hit'].play()
            #SOUNDS['die'].play()
            terminal = True
            self.__init__()
            reward = -1

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for u, l in zip(self.uppers, self.lowers):
            # SCREEN.blit(IMAGES['barrier'][0], (u['x'], u['y']))
            SCREEN.blit(IMAGES['barrier'][l['type']], (l['x'], l['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))

        showScore(self.score)
        # print("SCORE: %d" % self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()

        #print ("FPS" , FPSCLOCK.get_fps())
        FPSCLOCK.tick(FPS)

        return image_data, reward, terminal



def getRandomBarrier():
    """
        return a randomly generated barrier
    """
    interval = [10, 15, 20] # random interval between velX
    index = random.randint(0, len(interval) - 1)
    X = SCREENWIDTH + interval[index]
    Y = 30

    bar_type = random.randint(0, 2) # three types of velX

    return [
        {'x': X, 'y': Y, 'type': bar_type},  # upper
        {'x': X, 'y': BASEY - BARRIER_HEIGHT[bar_type], 'type': bar_type}, # lower
    ]


def showScore(score):
    """
        displays score in center of screen
    """
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) - 20

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, 20))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, uppers, lowers):
    """
        return True if player collides with barriers.
    """
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()
    playerRect = pygame.Rect(player['x'], player['y'], player['w'], player['h'])

    for u, l in zip(uppers, lowers):
        # upper and lower barrier rects
        uRect = pygame.Rect(u['x'], u['y'], BARRIER_WIDTH, BARRIER_HEIGHT[u['type']])
        lRect = pygame.Rect(l['x'], l['y'], BARRIER_WIDTH, BARRIER_HEIGHT[l['type']])

        # player and upper/lower barrier hitmasks
        pHitMask = HITMASKS['player'][pi]
        uHitmask = HITMASKS['barrier'][u['type']]
        lHitmask = HITMASKS['barrier'][l['type']]

        # if bird collided with u or l
        uCollide = pixelCollision(playerRect, uRect, pHitMask, uHitmask)
        lCollide = pixelCollision(playerRect, lRect, pHitMask, lHitmask)

        # if lCollide:
        if lCollide: return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """
        Check if two objects collide and not just their rects
    """

    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False
