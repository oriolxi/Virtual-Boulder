        point = [int(event.position().x()), int(event.position().y())]
        if point[0] is None or point[1] is None: return
        collision = self.__isPointInsideAnyRectangle(point)
        if collision is None: return
        idx = self.stored_points.index(collision)

        move = [self.holds[idx][0], self.holds[idx][1], self.holds[idx][2], self.holds[idx][3], self.wc]
        if event.button() == Qt.MouseButton.LeftButton:
            
            if move not in self.boulder:
                move[4] = "R"
                self.boulder.append(move)
            else:
                idx = self.boulder.index(move)
                if self.boulder[idx][4] == "L": self.boulder[idx][4] = "R"
                elif self.boulder[idx][4] == "R": self.boulder[idx][4] = "LR"
                elif self.boulder[idx][4] == "LR": self.boulder[idx][4] = "L"
        if event.button() == Qt.MouseButton.RightButton:
            if move in self.boulder:
                self.boulder.remove(move)

        self.__paintBoulder()