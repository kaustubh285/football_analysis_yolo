from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self) -> None:
        self.team_colors = {}
        self.player_team_dict = {}

    def get_cropped_image(self, frame, bbox):
        cropped_img = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        return cropped_img

    def get_clustering_model(self, image):
        # ? Too much background color, hence we cluster the image into 2 cluster

        # Reshape into 2d array
        image_2d = image.reshape(-1, 3)

        return KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)

    def get_player_color(self, frame, bbox):
        image = self.get_cropped_image(frame, bbox)

        image_top_half = image[0 : int(image.shape[0] / 2), :]

        kmeans = self.get_clustering_model(image_top_half)

        # Get labels
        labels = kmeans.labels_

        # Reshape the labels into the original image shape
        clustered_image = labels.reshape(
            image_top_half.shape[0], image_top_half.shape[1]
        )

        # ? We dont know what the labels for background or t-shirt are out of the two unless we plot it. Hence we check the corners becuase in most cases, the corners will have the background
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        for _, player in player_detections.items():
            bbox = player["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        km = KMeans(n_clusters=2, init="k-means++", n_init=1)
        km.fit(player_colors)

        self.km = km

        self.team_colors[1] = km.cluster_centers_[0]
        self.team_colors[2] = km.cluster_centers_[1]

    def assign_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, bbox)
        team_id = self.km.predict(player_color.reshape(1, -1))[0]

        # The above will return 0 or 1, but we've set the team colors as 1 or 2, hence we just add 1 to whatever we get
        team_id += 1
        self.player_team_dict[player_id] = team_id

        return team_id
