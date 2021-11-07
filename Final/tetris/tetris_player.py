import pygame
import random
import numpy as np
import time
import pickle

# name = input("Name: ").strip()
name = "test"

width = 780
height = 610
fps = 1000

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

# pygame 구동 관련 변수
run = True
GameStart = False
GameOver = False
Quit = False
hold = np.zeros((6, 2), int)


# 게임 내 변수
class inputs:
    moveLeft = False
    moveRight = False
    softDrop = False

    hardDrop = False
    rotateLeft = False
    rotateRight = False
    hold = False

    DAS_LEFT = False
    DAS_RIGHT = False

    SD_ARR_CNT = 0
    L_DAS_CNT = 0
    L_ARR_CNT = 0
    R_DAS_CNT = 0
    R_ARR_CNT = 0
    DROP_CNT = 0
    DOWN_CNT = 0
    LEVEL = 0

    # 조정 가능한 설정값
    LINE_INTERVAL = 10
    NEXT_LINE = LINE_INTERVAL
    END_GAME = 200

    SD_ARR_VALUE = 5
    DAS_VALUE = 30
    ARR_VALUE = 10
    DROP_VALUE = 100
    DOWN_VALUE = 5 * (1.5**10)


# 테트리스 보드
class field:
    matrix = np.zeros((40, 10), int)
    testMatrix = np.zeros((40, 10), int)

    def __init__(self):
        self.clearCnt = 0
        self.tspin = False

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
class mino:
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
            field.matrix[self.data[i, 1] + self.data[5, 1], self.data[i, 0] + self.data[5, 0]] = self.data[4, 0]

    # 필드에서 현재 미노를 지움
    def eraseMino(self):
        for i in range(4):
            field.matrix[self.data[i, 1] + self.data[5, 1], self.data[i, 0] + self.data[5, 0]] = 0

    # 미노가 움직이려고 하는 자리가 막혀있는지 확인
    # 왜인지 막혀있으면 False고 안 막히면 True임
    def isBlockedByMovement(self, toX, toY):
        x, y = toX + self.data[5, 0], toY + self.data[5, 1]
        for i in range(4):
            if not ((self.data[i, 0] + x) in range(10) and (self.data[i, 1] + y) in range(40) and field.testMatrix[
                    self.data[i, 1] + y, self.data[i, 0] + x] == 0):
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
        while (self.isBlockedByMovement(0, 1)):
            self.moveMino(0, 1)
        self.ghost = self.data.copy()
        self.data = self.testdata.copy()

    # 미노를 하드드롭함
    def hardDrop(self):
        self.eraseMino()
        self.data = self.ghost.copy()
        self.drawMino()


# 다음에 나올 미노가 담겨 있는 가방
class bag:
    def __init__(self):
        self.nowQueue = random.sample(mino.minoData, 7)

    def generateBag(self):
        if len(self.nowQueue) < 10:
            self.nowQueue += random.sample(mino.minoData, 7)


# 필드, 가방, 첫 미노 생성
f = field()
nowBag = bag()
nowMino = mino(nowBag.nowQueue.pop(0))
nowMino.drawMino()
while run and not GameOver:  # 게임 오버가 아니고, 작동하는 동안
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # 나가기 버튼을 누르면 게임 멈춤
            run = False
            Quit = True
        if event.type == pygame.KEYDOWN:  # 키보드가 눌리면...

            if event.key != pygame.K_RETURN and not GameStart:
                continue
            else:
                GameStart = True  # 엔터 키가 눌리고 게임이 시작하기 전이면 게임을 시작함

            if event.key == pygame.K_SPACE:  # 하드드롭
                nowMino.hardDrop()
                nowMino = mino(nowBag.nowQueue.pop(0))
                nowBag.generateBag()
                field.testMatrix = field.matrix.copy()
                nowMino.drawMino()
                f.clearLines()
                inputs.hardDrop = True
                inputs.hold = False

            if (event.key in [pygame.K_UP, pygame.K_x, pygame.K_w, pygame.K_KP8]) and nowMino.isSRS(1):  # 시계방향 회전
                nowMino.rotateMino(1)
                nowMino.drawMino()
                inputs.DROP_CNT = 0

            if (event.key in [pygame.K_z, pygame.K_q]) and nowMino.isSRS(-1):  # 반시계방향 회전
                nowMino.rotateMino(-1)
                nowMino.drawMino()
                inputs.DROP_CNT = 0

            # if event.key == pygame.K_x and nowMino.isSRS(-2):
            #     nowMino.rotateMino(-2)
            #     nowMino.drawMino()

            if (event.key in [pygame.K_c, pygame.K_e]) and not inputs.hold:  # 홀드
                nowMino.eraseMino()
                nowMino.holdMino()
                nowMino.drawMino()
                inputs.hold = True

            if event.key in [pygame.K_LEFT, pygame.K_KP4]:  # 왼쪽 이동 시작
                if nowMino.isBlockedByMovement(-1, 0):
                    nowMino.eraseMino()
                    nowMino.moveMino(-1, 0)
                    nowMino.drawMino()
                    inputs.DROP_CNT = 0
                inputs.moveLeft = True

            if event.key in [pygame.K_RIGHT, pygame.K_KP6]:  # 오른쪽 이동 시작
                if nowMino.isBlockedByMovement(1, 0):
                    nowMino.eraseMino()
                    nowMino.moveMino(1, 0)
                    nowMino.drawMino()
                    inputs.DROP_CNT = 0
                inputs.moveRight = True

            if (event.key in [pygame.K_DOWN, pygame.K_KP5]) and nowMino.isBlockedByMovement(0, 1):  # 소프트드롭 시작
                inputs.softDrop = True

        if event.type == pygame.KEYUP:

            if event.key in [pygame.K_LEFT, pygame.K_KP4]:  # 왼쪽 이동 중지
                inputs.L_DAS_CNT = 0
                inputs.L_ARR_CNT = 0
                inputs.moveLeft = False

            if event.key in [pygame.K_RIGHT, pygame.K_KP6]:  # 오른쪽 이동 중지
                inputs.R_DAS_CNT = 0
                inputs.R_ARR_CNT = 0
                inputs.moveRight = False

            if event.key in [pygame.K_DOWN, pygame.K_KP5]:  # 소프트드롭 중지
                inputs.SD_ARR_CNT = 0
                inputs.softDrop = False

    if inputs.hardDrop:
        inputs.hardDrop = False

    if inputs.moveLeft:  # 실제 왼쪽 이동

        if inputs.L_DAS_CNT >= inputs.DAS_VALUE and not inputs.moveRight:
            if inputs.ARR_VALUE == 0:
                nowMino.eraseMino()
                while nowMino.isBlockedByMovement(-1, 0):
                    nowMino.moveMino(-1, 0)
                    inputs.DROP_CNT = 0
                nowMino.drawMino()
            elif inputs.L_ARR_CNT % inputs.ARR_VALUE == 0 and nowMino.isBlockedByMovement(-1, 0):
                nowMino.eraseMino()
                nowMino.moveMino(-1, 0)
                inputs.DROP_CNT = 0
                nowMino.drawMino()
            inputs.L_ARR_CNT += 1
        inputs.L_DAS_CNT += 1

    if inputs.moveRight:  # 실제 오른쪽 이동

        if inputs.R_DAS_CNT >= inputs.DAS_VALUE and not inputs.moveLeft:
            if inputs.ARR_VALUE == 0:
                nowMino.eraseMino()
                while nowMino.isBlockedByMovement(1, 0):
                    nowMino.moveMino(1, 0)
                    inputs.DROP_CNT = 0
                nowMino.drawMino()
            elif inputs.R_ARR_CNT % inputs.ARR_VALUE == 0 and nowMino.isBlockedByMovement(1, 0):
                nowMino.eraseMino()
                nowMino.moveMino(1, 0)
                inputs.DROP_CNT = 0
                nowMino.drawMino()
            inputs.R_ARR_CNT += 1
        inputs.R_DAS_CNT += 1

    if inputs.softDrop:  # 실제 소프트드롭

        if inputs.SD_ARR_CNT % inputs.SD_ARR_VALUE == 0 and nowMino.isBlockedByMovement(0, 1):
            nowMino.eraseMino()
            nowMino.moveMino(0, 1)
            nowMino.drawMino()
        inputs.SD_ARR_CNT += 1

    if GameStart:  # 게임이 구동중인 동안...

        if inputs.DOWN_CNT < inputs.DOWN_VALUE:
            inputs.DOWN_CNT += 1
        else:
            if nowMino.isBlockedByMovement(0, 1):  # 자동 떨어짐
                nowMino.eraseMino()
                nowMino.moveMino(0, 1)
                nowMino.drawMino()
            inputs.DOWN_CNT = 0

    if not nowMino.isBlockedByMovement(0, 1):  # 바닥 상태일 때(?) 자동 드롭

        if inputs.DROP_CNT < inputs.DROP_VALUE:
            inputs.DROP_CNT += 1
        else:
            nowMino.hardDrop()
            nowMino = mino(nowBag.nowQueue.pop(0))
            nowBag.generateBag()
            field.testMatrix = field.matrix.copy()
            nowMino.drawMino()
            f.clearLines()
            inputs.hold = False
            inputs.DROP_CNT = 0
    else:
        inputs.DROP_CNT = 0

    nowMino.drawGhost()
    screen.fill(white)

    for i in range(4):  # 떨어질 위치 보여줌
        pygame.draw.rect(screen, lightgray, ((nowMino.ghost[i, 0] + nowMino.ghost[5, 0]) * 30 + 241,
                                             (nowMino.ghost[i, 1] + nowMino.ghost[5, 1] - 19) * 30 + 1 - 20, 28, 28))
        pygame.draw.rect(screen, white, ((nowMino.ghost[i, 0] + nowMino.ghost[5, 0]) * 30 + 245,
                                         (nowMino.ghost[i, 1] + nowMino.ghost[5, 1] - 19) * 30 + 5 - 20, 20, 20))

    for idx_i, val_i in enumerate(field.matrix):  # 아마 다른 타일 그리는 건 듯
        for idx_j, val_j in enumerate(val_i):
            if field.matrix[idx_i - 21, idx_j] != 0:
                pygame.draw.rect(screen, cell_Colors[field.matrix[idx_i - 21, idx_j]],
                                 (idx_j * 30 + 241, idx_i * 30 + 1 - 20, 28, 28))

    for i in range(40):  # 배경 그리는 건 듯
        for j in range(26):
            if j < 8 or j >= 18:
                pygame.draw.rect(screen, lightgray, (j * 30, i * 30 - 600 + 10, 30, 30), 1)
            else:
                pygame.draw.rect(screen, (230, 230, 230), (j * 30, i * 30 - 600 + 10, 30, 30), 1)

    for i in range(5):  # 다음 블록 보여줌
        mino.nexts[i] = nowBag.nowQueue[i]
        for j in range(4):
            pygame.draw.rect(screen, cell_Colors[mino.nexts[i][4, 0]], (
                570 + 1 + mino.nexts[i][j, 0] * 30, 60 + 1 - 20 + mino.nexts[i][j, 1] * 30 + i * 90, 28, 28))
            pygame.draw.rect(screen, cell_Colors[hold[4, 0]],
                             (60 + 1 + hold[j, 0] * 30, 60 + 1 - 20 + hold[j, 1] * 30, 28, 28))

    if not GameStart:  # 게임 시작하기 전 시작 문구
        screen.blit(sText, text_rect)

    if f.clearCnt >= inputs.NEXT_LINE and inputs.LEVEL < 10:  # 목표 줄 수 넘기면 레벨업
        inputs.NEXT_LINE += inputs.LINE_INTERVAL
        inputs.LEVEL += 1
        inputs.DROP_VALUE -= 3
        inputs.DOWN_VALUE /= 1.5

    elif f.clearCnt >= inputs.END_GAME:  # 엔드 게임 줄 수 넘기면 엔드 게임 시작?!
        inputs.LEVEL = -1
        inputs.DROP_VALUE = 70
        inputs.DOWN_VALUE = 2

    # 스코어 텍스트 보여주기
    score_text = pygame.font.SysFont("arial", 32, True, False).render(f"{f.clearCnt}000", True, black)
    score_rect = score_text.get_rect()
    score_rect.x = 570
    score_rect.y = 486
    screen.blit(score_text, score_rect)

    # 레벨 텍스트 보여주기
    score_text = pygame.font.SysFont("arial", 28, True, False).render(f"LEVEL: {inputs.LEVEL}", True, black)
    score_rect = score_text.get_rect()
    score_rect.x = 574
    score_rect.y = 518
    screen.blit(score_text, score_rect)

    pygame.display.flip()
    clock.tick(fps)

# 게임이 끝남
if GameStart and not Quit:  # 게임 강종이 아니라면

    # 게임 오버 출력
    mFont = pygame.font.SysFont("arial", 50, True, False)
    sText = mFont.render("Game Over!", True, black)
    text_rect = sText.get_rect()
    text_rect.centerx = round(width / 2)
    text_rect.centery = round(height / 2)
    screen.blit(sText, text_rect)
    pygame.display.flip()
    clock.tick(fps)

    time.sleep(2)

    # 기록 호출 및 리더보드 출력
    try:
        with open("score.txt", "rb") as g:
            scores = pickle.load(g)
    except:
        scores = []

    scores.append((name, f.clearCnt * 1000))
    scores.sort(key=lambda x: x[1], reverse=True)
    screen.fill(white)
    sText = mFont.render("Highscore:", True, black)
    text_rect = sText.get_rect()
    text_rect.centerx = round(width / 2)
    text_rect.centery = 30
    screen.blit(sText, text_rect)

    j = 100
    for i in scores[:8]:
        text = f"{i[0]}: {i[1]}"
        sText = mFont.render(text, True, black)
        text_rect = sText.get_rect()
        text_rect.centerx = round(width / 2)
        text_rect.centery = j
        j += 70
        screen.blit(sText, text_rect)
    pygame.display.flip()
    clock.tick(fps)

    with open("score.txt", "wb") as g:
        pickle.dump(scores, g)

    time.sleep(5)

pygame.quit()
