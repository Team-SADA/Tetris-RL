import sys
import pygame
import random
import numpy as np
import torch

width = 780
height = 610
fps = 100000

# 색상 선언
white = (255, 255, 255)
cyan = (0, 255, 255)
yellow = (255, 255, 0)
mazenta = (255, 0, 255)
orange = (255, 127, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
red = (255, 0, 0)
gray = (127, 127, 127)
lightgray = (180, 180, 180)
black = (0, 0, 0)
cell_Colors = [white, cyan, yellow, mazenta, orange, blue, green, red, gray, black]

# pygame 설정
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pytris")
clock = pygame.time.Clock()
mFont = pygame.font.SysFont("arial", 50, True, False)
sText = mFont.render("Press Enter", True, black)
text_rect = sText.get_rect()
text_rect.centerx = round(width / 2)
text_rect.centery = round(height / 2)
# 246, 990
a = random.randint(1, 1000)
random.seed(990)
print(a)


# 게임 내 변수
class Inp:
    def __init__(self):
        self.DROP_CNT = 0
        self.DOWN_CNT = 0

        self.DROP_VALUE = 100
        self.DOWN_VALUE = 75

        self.LEVEL = 0
        self.NEXT_LINE = 10
        self.END_GAME = 200


# 테트리스 보드
class Field:

    def __init__(self):
        self.clearCnt = 0
        self.tspin = False
        self.matrix = np.zeros((40, 10), int)
        self.testMatrix = np.zeros((40, 10), int)
        self.temp_matrix = np.zeros((40, 10), int)

    # 라인 삭제 함수
    def clearLines(self):
        for i in range(20):
            if np.all(self.testMatrix[20 + i]):
                self.matrix[20 + i] = 0
                self.testMatrix[20 + i] = 0
                self.matrix[:20 + i + 1] = np.roll(self.testMatrix[:20 + i + 1], 1, axis=0)
                self.testMatrix[:20 + i + 1] = np.roll(self.testMatrix[:20 + i + 1], 1, axis=0)
                self.clearCnt += 1
                nowMino.drawMino()


# 테트리미노 블록
class Mino:
    I = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [1, 0], [3, 19]])
    O = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [4, 19]])
    T = np.array([[0, 1], [1, 0], [1, 1], [2, 1], [3, 0], [3, 19]])
    L = np.array([[0, 1], [1, 1], [2, 0], [2, 1], [4, 0], [3, 19]])
    J = np.array([[0, 0], [0, 1], [1, 1], [2, 1], [5, 0], [3, 19]])
    S = np.array([[0, 1], [1, 0], [1, 1], [2, 0], [6, 0], [3, 19]])
    Z = np.array([[0, 0], [1, 0], [1, 1], [2, 1], [7, 0], [3, 19]])
    X = np.zeros((6, 2), int)
    minoData = [I, O, T, L, J, S, Z]
    rotMat = np.array([[[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]]])
    SRS2 = np.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                     [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                     [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                     [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]])
    SRS3 = np.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                     [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
                     [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                     [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]]])
    SRS4 = np.array([[[0, 0], [-1, 0], [2, 0], [-1, 0], [2, 0]],
                     [[0, 0], [1, 0], [1, 0], [1, -1], [1, 2]],
                     [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, 1]],
                     [[0, 0], [0, 0], [0, 0], [0, 2], [0, -1]]])
    SRS = np.array([SRS2, SRS3, SRS4])
    nexts = [0, 0, 0, 0, 0]

    # 새로운 미노를 만드는 듯
    def __init__(self, newMino):
        global GameOver
        self.data = newMino.copy()
        self.testdata = self.data.copy()
        self.ghost = self.data.copy()
        if (f.matrix[19][3:7] + f.matrix[20][3:7]).any():
            GameOver = True

    # 현재 미노를 제거하는 듯
    def __del__(self):
        if self.data[4, 1] != 0:
            for i in range(self.data[4, 1]):
                self.rotateMino(-1)

    # 필드에 현재 미노를 그림
    def drawMino(self):
        for i in range(4):
            f.matrix[self.data[i, 1] + self.data[5, 1], self.data[i, 0] + self.data[5, 0]] = self.data[4, 0]

    # 필드에서 현재 미노를 지움
    def eraseMino(self):
        for i in range(4):
            f.matrix[self.data[i, 1] + self.data[5, 1], self.data[i, 0] + self.data[5, 0]] = 0

    # 미노가 움직이려고 하는 자리가 막혀있는지 확인
    # 왜인지 막혀있으면 False고 안 막히면 True임
    def isBlockedByMovement(self, toX, toY):
        x, y = toX + self.data[5, 0], toY + self.data[5, 1]
        for i in range(4):
            if not ((self.data[i, 0] + x) in range(10) and (self.data[i, 1] + y) in range(40)
                    and f.testMatrix[self.data[i, 1] + y, self.data[i, 0] + x] == 0):
                return False
        return True

    # 미노를 정해진 대로 움직임
    def moveMino(self, toX, toY):
        self.data[5, 0] += toX
        self.data[5, 1] += toY

    # 테트리스 회전 규칙을 적용시킴
    def isSRS(self, d):
        self.eraseMino()
        self.testdata = self.data.copy()
        ad = self.data[4, 1].copy()
        self.rotateMino(d)
        bd = self.data[4, 1].copy()
        for i in range(5):
            if self.isBlockedByMovement(
                    self.SRS[np.max(self.minoData[self.data[4, 0] - 1][:4]) - 1, ad, i, 0] - self.SRS[
                        np.max(self.minoData[self.data[4, 0] - 1][:4]) - 1, bd, i, 0],
                    self.SRS[np.max(self.minoData[self.data[4, 0] - 1][:4]) - 1, ad, i, 1] - self.SRS[
                        np.max(self.minoData[self.data[4, 0] - 1][:4]) - 1, bd, i, 1]):
                self.moveMino(self.SRS[np.max(self.minoData[self.data[4, 0] - 1][:4]) - 1, ad, i, 0] - self.SRS[
                    np.max(self.minoData[self.data[4, 0] - 1][:4]) - 1, bd, i, 0],
                              self.SRS[np.max(self.minoData[self.data[4, 0] - 1][:4]) - 1, ad, i, 1] - self.SRS[
                                  np.max(self.minoData[self.data[4, 0] - 1][:4]) - 1, bd, i, 1])
                self.data[:5] = self.testdata[:5].copy()
                return True
        self.data[:5] = self.testdata[:5].copy()
        self.drawMino()
        return False

    # 미노를 실제로 회전시킴
    def rotateMino(self, d):
        self.data[4, 1] = (self.data[4, 1] + d + 4) % 4
        for i in range(4):
            self.data[i] = np.dot(self.rotMat[d], [
                self.data[i, 0] + np.max(self.minoData[self.data[4, 0] - 1][:4]) * np.min(self.rotMat[d, 1]),
                self.data[i, 1] + np.max(self.minoData[self.data[4, 0] - 1][:4]) * np.min(self.rotMat[d, 0])])

    # 미노를 저장(홀드)시킴
    def holdMino(self):
        global hold
        self.data[5, 0] = self.minoData[self.data[4, 0] - 1][5, 0].copy()
        self.data[5, 1] = self.minoData[self.data[4, 0] - 1][5, 1].copy()
        if self.data[4, 1] != 0:
            for i in range(self.data[4, 1]):
                self.rotateMino(-1)
        self.data[4, 1] = 0
        if hold[4, 0] == self.X[4, 0]:
            hold = self.data
            self.data = nowBag.nowQueue.pop(0).copy()
        else:
            hold, self.data = self.data.copy(), hold.copy()

    # 미노의 그림자를 그림
    def drawGhost(self):
        self.testdata = self.data.copy()
        while self.isBlockedByMovement(0, 1):
            self.moveMino(0, 1)
        self.ghost = self.data.copy()
        self.data = self.testdata.copy()

    # 미노를 하드드롭함
    def hardDrop(self):
        self.eraseMino()
        self.data = self.ghost.copy()
        self.drawMino()


# 다음에 나올 미노가 담겨 있는 가방
class Bag:
    def __init__(self):
        self.nowQueue = random.sample(Mino.minoData, 7)

    def generateBag(self):
        if len(self.nowQueue) < 10:
            self.nowQueue += random.sample(Mino.minoData, 7)


# pygame 구동 관련 변수
run = True
GameStart = True
GameOver = False
Quit = False

inputs = Inp()
f = Field()
nowBag = Bag()
nowMino = Mino(nowBag.nowQueue[0])
nowMino.drawMino()
hold = np.zeros((6, 2), int)
score = 0
tetromino = 0


def reset():
    global inputs, f, nowBag, nowMino, GameOver, run, hold, score, tetromino
    inputs = Inp()
    f = Field()
    nowBag = Bag()
    nowMino = Mino(nowBag.nowQueue.pop(0))
    nowMino.drawMino()
    GameOver = False
    run = True
    hold = np.zeros((6, 2), int)
    score = 0
    tetromino = 0

    return get_state_properties(f.matrix)


def clear_line(board):
    to_delete = []
    for i, row in enumerate(board):
        if 0 not in row:
            to_delete.append(len(board) - 1 - i)
    if len(to_delete) > 0:
        board = remove_row(board, to_delete)
    return len(to_delete), board


def remove_row(board, to_delete):
    for i in to_delete[::-1]:
        board = list(board)
        del board[i]
        board = np.array(board)
        board = [[0 for _ in range(10)]] + board
    return board


def get_holes(board):
    num_holes = 0
    for col in zip(*board):
        row = 0
        while row < len(col):
            if col[row] != 0:
                break
            row += 1
        num_holes += len([x for x in col[row + 1:] if x == 0])
    return num_holes


def get_bumpiness_and_height(board):
    board = np.array(board)
    mask = board != 0
    invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), len(board))
    heights = len(board) - invert_heights
    total_height = np.sum(heights)
    currs = heights[:-1]
    nexts = heights[1:]
    diffs = np.abs(currs - nexts)
    total_bumpiness = np.sum(diffs)
    return total_bumpiness, total_height


def get_state_properties(board):
    lines_cleared, board = clear_line(board)
    holes = get_holes(board)
    bumpiness, height = get_bumpiness_and_height(board)
    return torch.FloatTensor([lines_cleared, holes, bumpiness, height])


def get_next_states():
    global nowMino

    states = {}

    f.temp_matrix = f.matrix.copy()
    not_mino = nowMino.data.copy()

    for rotate in range(4):
        for x in range(-5, 6):
            dx = x
            f.matrix = f.temp_matrix.copy()
            nowMino.data = not_mino.copy()

            if rotate == 1 and nowMino.isSRS(1):
                nowMino.rotateMino(1)
                nowMino.drawMino()
            if rotate == 2 and nowMino.isSRS(-2):
                nowMino.rotateMino(-2)
                nowMino.drawMino()
            if rotate == 3 and nowMino.isSRS(-1):
                nowMino.rotateMino(-1)
                nowMino.drawMino()

            while x != 0:
                if x > 0 and nowMino.isBlockedByMovement(1, 0):
                    nowMino.eraseMino()
                    nowMino.moveMino(1, 0)
                    nowMino.drawMino()
                if x < 0 and nowMino.isBlockedByMovement(-1, 0):
                    nowMino.eraseMino()
                    nowMino.moveMino(-1, 0)
                    nowMino.drawMino()
                x += -1 if x > 0 else 1

            nowMino.eraseMino()
            while nowMino.isBlockedByMovement(0, 1):
                nowMino.moveMino(0, 1)
            nowMino.drawMino()

            states[(rotate, dx)] = get_state_properties(f.matrix)

    f.matrix = f.temp_matrix.copy()
    nowMino.data = not_mino.copy()
    return states


def step(action):
    global run, GameOver, GameStart, Quit, nowMino, nowBag, hold, f, inputs, score, tetromino

    is_action = True
    old_cnt = f.clearCnt

    while is_action:  # 취할 액션이 없을 때까지 반복

        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # 나가기 버튼을 누르면 게임 멈춤
                sys.exit()

        if action[0]:
            if action[0] == 1 and nowMino.isSRS(1):
                nowMino.rotateMino(1)
                nowMino.drawMino()
            if action[0] == 2 and nowMino.isSRS(-2):
                nowMino.rotateMino(-2)
                nowMino.drawMino()
            if action[0] == 3 and nowMino.isSRS(-1):
                nowMino.rotateMino(-1)
                nowMino.drawMino()
            action[0] = 0
        elif action[1]:
            if action[1] > 0 and nowMino.isBlockedByMovement(1, 0):
                nowMino.eraseMino()
                nowMino.moveMino(1, 0)
                nowMino.drawMino()
            if action[1] < 0 and nowMino.isBlockedByMovement(-1, 0):
                nowMino.eraseMino()
                nowMino.moveMino(-1, 0)
                nowMino.drawMino()
            action[1] += -1 if action[1] > 0 else 1
        else:
            nowMino.hardDrop()
            nowMino = Mino(nowBag.nowQueue.pop(0))
            nowBag.generateBag()
            f.testMatrix = f.matrix.copy()
            nowMino.drawMino()
            f.clearLines()

            is_action = False

        if inputs.DOWN_CNT < inputs.DOWN_VALUE:
            inputs.DOWN_CNT += 1
        else:
            if nowMino.isBlockedByMovement(0, 1):  # 자동 떨어짐
                nowMino.eraseMino()
                nowMino.moveMino(0, 1)
                nowMino.drawMino()
            inputs.DOWN_CNT = 0

        nowMino.drawGhost()
        screen.fill(white)

        for i in range(4):  # 떨어질 위치 보여줌
            pygame.draw.rect(screen, lightgray, ((nowMino.ghost[i, 0] + nowMino.ghost[5, 0]) * 30 + 241,
                                                 (nowMino.ghost[i, 1] + nowMino.ghost[5, 1] - 19) * 30 + 1 - 20, 28,
                                                 28))
            pygame.draw.rect(screen, white, ((nowMino.ghost[i, 0] + nowMino.ghost[5, 0]) * 30 + 245,
                                             (nowMino.ghost[i, 1] + nowMino.ghost[5, 1] - 19) * 30 + 5 - 20, 20, 20))

        for idx_i, val_i in enumerate(f.matrix):  # 아마 다른 타일 그리는 건 듯
            for idx_j, val_j in enumerate(val_i):
                if f.matrix[idx_i - 21, idx_j] != 0:
                    pygame.draw.rect(screen, cell_Colors[f.matrix[idx_i - 21, idx_j]],
                                     (idx_j * 30 + 241, idx_i * 30 + 1 - 20, 28, 28))

        for i in range(40):  # 배경 그리는 건 듯
            for j in range(26):
                if j < 8 or j >= 18:
                    pygame.draw.rect(screen, lightgray, (j * 30, i * 30 - 600 + 10, 30, 30), 1)
                else:
                    pygame.draw.rect(screen, (230, 230, 230), (j * 30, i * 30 - 600 + 10, 30, 30), 1)

        for i in range(5):  # 다음 블록 보여줌
            Mino.nexts[i] = nowBag.nowQueue[i]
            for j in range(4):
                pygame.draw.rect(screen, cell_Colors[Mino.nexts[i][4, 0]], (
                    570 + 1 + Mino.nexts[i][j, 0] * 30, 60 + 1 - 20 + Mino.nexts[i][j, 1] * 30 + i * 90, 28, 28))
                pygame.draw.rect(screen, cell_Colors[hold[4, 0]],
                                 (60 + 1 + hold[j, 0] * 30, 60 + 1 - 20 + hold[j, 1] * 30, 28, 28))

        if f.clearCnt >= inputs.NEXT_LINE and inputs.LEVEL < 10:  # 목표 줄 수 넘기면 레벨업
            inputs.NEXT_LINE += 10
            inputs.LEVEL += 1
            inputs.DROP_VALUE -= 3
            inputs.DOWN_VALUE -= 7

        elif f.clearCnt >= inputs.END_GAME:  # 엔드 게임 줄 수 넘기면 엔드 게임 시작?!
            inputs.LEVEL = -1
            inputs.DROP_VALUE = 70
            inputs.DOWN_VALUE = 0

        # 스코어 텍스트 보여주기
        score_text = pygame.font.SysFont("arial", 32, True, False).render(str(score), True, black)
        score_rect = score_text.get_rect()
        score_rect.x = 570
        score_rect.y = 486
        screen.blit(score_text, score_rect)

        score_text = pygame.font.SysFont("arial", 28, True, False).render(f"LINES: {f.clearCnt}", True, black)
        score_rect = score_text.get_rect()
        score_rect.x = 580
        score_rect.y = 518
        screen.blit(score_text, score_rect)

        # 레벨 텍스트 보여주기
        score_text = pygame.font.SysFont("arial", 28, True, False).render(f"LEVEL: {inputs.LEVEL}", True, black)
        score_rect = score_text.get_rect()
        score_rect.x = 574
        score_rect.y = 550
        screen.blit(score_text, score_rect)

        pygame.display.flip()
        clock.tick(fps)

    score_now = 10 * ((f.clearCnt - old_cnt) ** 2) + 1
    tetromino += 1

    score += score_now
    return score_now, GameOver
