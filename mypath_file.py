class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = "..."
            # Save preprocess data into output_dir
            output_dir = "..."

            return root_dir, output_dir
        elif database == 'cardiac_vids':
            # folder that contains VIdeos
            root_dir = "..."

            # folder that contains Videos Frames
            output_dir = "..."

            return root_dir, output_dir
        elif database == 'cardiac_mini_vids':
		# folder that contains VIdeos
            root_dir = "..."
            # folder that contains Videos Frames
            output_dir = "..."

        return root_dir, output_dir


    @staticmethod
    def model_dir():
        return 'c3d-pretrained.pth'