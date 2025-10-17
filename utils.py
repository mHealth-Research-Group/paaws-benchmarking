"""
=========================================
Activity regrouping schemes and lists of datasets we used in our experiments.
=========================================
Author: Veronika K. Potter
Email: potter[dot]v[at]northeastern[dot]edu (potter.v@northeastern.edu)
"""


# The mapping schemes used across our experiments.

# * Keys: string
#     The mapping scheme name.
# * Values: dict of string
#     A dict containing the mapping of raw labels to desired activity
#     groupings.
MAPPING_SCHEMES = {
    # Mapping with 5 activities, can be used with SimFL+Lab or FL data.
    "lab_fl_5": {
        "Sitting_Still": "Sitting",
        "Sitting_With_Movement": "Sitting",
        "Sit_Recline_Talk_Lab": "Sitting",
        "Sit_Recline_Web_Browse_Lab": "Sitting",
        "Sit_Writing_Lab": "Sitting",
        "Sit_Typing_Lab": "Sitting",

        "Standing_Still": "Standing",
        "Standing_With_Movement": "Standing",
        "Stand_Conversation_Lab": "Standing",

        "Lying_Still": "Lying_Down",
        "Lying_With_Movement": "Lying_Down",
        "Lying_On_Back_Lab": "Lying_Down",
        "Lying_On_Right_Side_Lab": "Lying_Down",
        "Lying_On_Stomach_Lab": "Lying_Down",
        "Lying_On_Left_Side_Lab": "Lying_Down",

        "Walking": "Walking",
        "Treadmill_2mph_Lab": "Walking",
        "Treadmill_3mph_Coversation_Lab": "Walking",
        "Treadmill_3mph_Free_Walk_Lab": "Walking",
        "Treadmill_3mph_Drink_Lab": "Walking",
        "Treadmill_3mph_Briefcase_Lab": "Walking",
        "Treadmill_3mph_Phone_Lab": "Walking",
        "Treadmill_3mph_Hands_Pockets_Lab": "Walking",
        "Walking_Fast": "Walking",
        "Walking_Slow": "Walking",

        'Stationary_Biking_300_Lab': 'Biking',
        'Exercising_Gym_Stationary_Bicycle': 'Biking',
        'Cycling_Active_Pedaling_Stationary_Bicycle': 'Biking',
        'Cycling_Active_Pedaling_Regular_Bicycle': 'Biking',
    },

    # Mapping with 9 activities, can be used with SimFL+Lab or FL data.
    "lab_fl_9": {
        "Sitting_Still": "Sitting",
        "Sitting_With_Movement": "Sitting",
        "Sit_Recline_Talk_Lab": "Sitting",
        "Sit_Recline_Web_Browse_Lab": "Sitting",
        "Sit_Writing_Lab": "Sitting",
        "Sit_Typing_Lab": "Sitting",

        "Standing_Still": "Standing",
        "Standing_With_Movement": "Standing",
        "Stand_Conversation_Lab": "Standing",

        "Lying_Still": "Lying_Down",
        "Lying_With_Movement": "Lying_Down",
        "Lying_On_Back_Lab": "Lying_Down",
        "Lying_On_Right_Side_Lab": "Lying_Down",
        "Lying_On_Stomach_Lab": "Lying_Down",
        "Lying_On_Left_Side_Lab": "Lying_Down",

        "Walking": "Walking",
        "Treadmill_2mph_Lab": "Walking",
        "Treadmill_3mph_Coversation_Lab": "Walking",
        "Treadmill_3mph_Free_Walk_Lab": "Walking",
        "Treadmill_3mph_Drink_Lab": "Walking",
        "Treadmill_3mph_Briefcase_Lab": "Walking",
        "Treadmill_3mph_Phone_Lab": "Walking",
        "Treadmill_3mph_Hands_Pockets_Lab": "Walking",
        "Walking_Fast": "Walking",
        "Walking_Slow": "Walking",

        "Walking_Up_Stairs": "Walking_Up_Stairs",

        "Walking_Down_Stairs": "Walking_Down_Stairs",

        "Ab_Crunches_Lab": "Gym_Exercises",
        "Arm_Curls_Lab": "Gym_Exercises",
        "Push_Up_Lab": "Gym_Exercises",
        "Push_Up_Modified_Lab": "Gym_Exercises",
        "Machine_Leg_Press_Lab": "Gym_Exercises",
        "Machine_Chest_Press_Lab": "Gym_Exercises",
        "Treadmill_5.5mph_Lab": "Gym_Exercises",

        "Cycling_Active_Pedaling_Regular_Bicycle": "Biking",
        "Stationary_Biking_300_Lab": "Biking",

        'Organizing_Shelf/Closet': 'Household_Chores',
        'Sweeping': 'Household_Chores',
        'Vacuuming': 'Household_Chores',
        'Stand_Shelf_Load_Lab': 'Household_Chores',
        'Stand_Shelf_Unload_Lab': 'Household_Chores',
        'Washing_Dishes_Lab': 'Household_Chores',
        'Chopping_Food_Lab': 'Household_Chores'
    },

    # Mapping with 42 activities, used with SimFL+Lab data only.
    "lab_42": {
        "Ab_Crunches_Lab": "Ab_Crunches_Lab",
        "Arm_Curls_Lab": "Arm_Curls_Lab",
        "Chopping_Food_Lab": "Chopping_Food_Lab",
        "Cycling_Active_Pedaling_Regular_Bicycle":
            "Cycling_Active_Pedaling_Regular_Bicycle",
        "Folding_Clothes": "Folding_Clothes",
        "Lying_On_Back_Lab": "Lying_On_Back_Lab",
        "Lying_On_Left_Side_Lab": "Lying_On_Left_Side_Lab",
        "Lying_On_Right_Side_Lab": "Lying_On_Right_Side_Lab",
        "Lying_On_Stomach_Lab": "Lying_On_Stomach_Lab",
        "Machine_Chest_Press_Lab": "Machine_Chest_Press_Lab",
        "Machine_Leg_Press_Lab": "Machine_Leg_Press_Lab",
        "Organizing_Shelf/Cabinet": "Organizing_Shelf/Cabinet",
        "Playing_Frisbee": "Playing_Frisbee",
        "Push_Up_Lab": "Push_Up_Lab",
        "Push_Up_Modified_Lab": "Push_Up_Modified_Lab",
        "Sit_Recline_Talk_Lab": "Sit_Recline_Talk_Lab",
        "Sit_Recline_Web_Browse_Lab": "Sit_Recline_Web_Browse_Lab",
        "Sit_Typing_Lab": "Sit_Typing_Lab",
        "Sit_Writing_Lab": "Sit_Writing_Lab",
        "Sitting_Still": "Sitting_Still",
        "Sitting_With_Movement": "Sitting_With_Movement",
        "Stand_Conversation_Lab": "Stand_Conversation_Lab",
        "Stand_Shelf_Load_Lab": "Stand_Shelf_Load_Lab",
        "Stand_Shelf_Unload_Lab": "Stand_Shelf_Unload_Lab",
        "Standing_Still": "Standing_Still",
        "Standing_With_Movement": "Standing_With_Movement",
        "Stationary_Biking_300_Lab": "Stationary_Biking_300_Lab",
        "Sweeping": "Sweeping",
        "Treadmill_2mph_Lab": "Treadmill_2mph_Lab",
        "Treadmill_3mph_Conversation_Lab": "Treadmill_3mph_Conversation_Lab",
        "Treadmill_3mph_Drink_Lab": "Treadmill_3mph_Drink_Lab",
        "Treadmill_3mph_Free_Walk_Lab": "Treadmill_3mph_Free_Walk_Lab",
        "Treadmill_3mph_Hands_Pockets_Lab": "Treadmill_3mph_Hands_Pockets_Lab",
        "Treadmill_3mph_Briefcase_Lab": "Treadmill_3mph_Briefcase_Lab",
        "Treadmill_3mph_Phone_Lab": "Treadmill_3mph_Phone_Lab",
        "Treadmill_4mph_Lab": "Treadmill_4mph_Lab",
        "Treadmill_5.5mph_Lab": "Treadmill_5.5mph_Lab",
        "Vacuuming": "Vacuuming",
        "Walking": "Walking",
        "Walking_Down_Stairs": "Walking_Down_Stairs",
        "Walking_Up_Stairs": "Walking_Up_Stairs",
        "Washing_Dishes_Lab": "Washing_Dishes_Lab",
    },

    # Mapping with 11 activities, used for FL data only.
    "fl_11": {
        'Standing_Still': 'Standing',
        'Stand_Conversation_Lab': 'Standing',
        'Standing_With_Movement': 'Standing',

        'Sitting_Still': 'Sitting',
        'Sit_Typing_Lab': 'Sitting',
        'Sit_Recline_Web_Browse_Lab': 'Sitting',
        'Sit_Writing_Lab': 'Sitting',
        'Sit_Recline_Talk_Lab': 'Sitting',
        'Sitting_With_Movement': 'Sitting',

        'Lying_Still': 'Lying_Down',
        'Lying_On_Left_Side_Lab': 'Lying_Down',
        'Lying_On_Right_Side_Lab': 'Lying_Down',
        'Lying_On_Back_Lab': 'Lying_Down',
        'Lying_On_Stomach_Lab': 'Lying_Down',
        'Lying_With_Movement': 'Lying_Down',

        'Walking': 'Walking',
        'Walking_Slow': 'Walking',
        'Walking_Fast': 'Walking',
        'Walking_Up_Stairs': 'Walking',
        'Walking_Down_Stairs': 'Walking',
        'Walking_Treadmill': 'Walking',
        'Treadmill_2mph_Lab': 'Walking',
        'Treadmill_3mph_Conversation_Lab': 'Walking',
        'Treadmill_3mph_Drink_Lab': 'Walking',
        'Treadmill_3mph_Free_Walk_Lab': 'Walking',
        'Treadmill_3mph_Briefcase_Lab': 'Walking',
        'Treadmill_3mph_Hands_Pockets_Lab': 'Walking',
        'Treadmill_4mph_Lab': 'Walking',

        'Stationary_Biking_300_Lab': 'Biking',
        'Exercising_Gym_Stationary_Bicycle': 'Biking',
        'Cycling_Active_Pedaling_Stationary_Bicycle': 'Biking',
        'Cycling_Active_Pedaling_Regular_Bicycle': 'Biking',

        'In_Transit_Driving_Car': 'Driving',

        'Playing_Frisbee': 'Exercising',
        'Playing_Exergame': 'Exercising',
        'Playing_Sports/Games': 'Exercising',
        'Doing_Resistance_Training_Free_Weights': 'Exercising',
        'Exercising_Gym_Other': 'Exercising',
        'Doing_Resistance_Training_Other': 'Exercising',
        'Exercising_Gym_Treadmill': 'Exercising',
        'Arm_Curls_Lab': 'Exercising',
        'Doing_Martial_Arts': 'Exercising',
        'Push_Up_Modified_Lab': 'Exercising',
        'Doing_Resistance_Training': 'Exercising',
        'Push_Up_Lab': 'Exercising',
        'Ab_Crunches_Lab': 'Exercising',
        'Machine_Leg_Press_Lab': 'Exercising',
        'Machine_Chest_Press_Lab': 'Exercising',
        'Running_Non_Treadmill': 'Exercising',
        'Running_Treadmill': 'Exercising',

        'Dry_Mopping': 'Household_Chores',
        'Cleaning': 'Household_Chores',
        'Dusting': 'Household_Chores',
        'Doing_Common_Housework_Light': 'Household_Chores',
        'Folding_Clothes': 'Household_Chores',
        'Doing_Common_Housework': 'Household_Chores',
        'Ironing': 'Household_Chores',
        'Doing_Dishes': 'Household_Chores',
        'Loading/Unloading_Washing_Machine/Dryer': 'Household_Chores',
        'Doing_Home_Repair_Light': 'Household_Chores',
        'Organizing_Shelf/Closet': 'Household_Chores',
        'Doing_Home_Repair': 'Household_Chores',
        'Putting_Clothes_Away': 'Household_Chores',
        'Packing/Unpacking_Something': 'Household_Chores',
        'Sweeping': 'Household_Chores',
        'Vacuuming': 'Household_Chores',
        'Stand_Shelf_Load_Lab': 'Household_Chores',
        'Stand_Shelf_Unload_Lab': 'Household_Chores',
        'Washing_Dishes_Lab': 'Household_Chores',

        'Chopping_Food_Lab': 'Cooking',
        'Cooking/Prepping_Food': 'Cooking',

        'Eating/Dining': 'Eating/Drinking',

        'Washing_Hands': 'Grooming',
        'Brushing_Teeth': 'Grooming',
        'Brushing/Combing/Tying_Hair': 'Grooming',
        'Washing_Face': 'Grooming',
        'Applying_Makeup': 'Grooming',
        'Flossing_Teeth': 'Grooming',
        'Blowdrying_Hair': 'Grooming',
    },
}


# The lists of participant IDs that we used in each experiment.
# * Keys: string
#     The set label for the experiment.
# * Values: list of int
#     A list of DS_IDs to be used in the the experiment.
DATASET_LISTS = {

    "2": [10, 36],  # For launching a test job in the repo.

    "SimFL_20": [10, 36, 37, 38, 39, 42, 44, 48, 49, 51, 58,
                 59, 87, 138, 139, 140, 235, 239, 240, 246],

    "FL_20": [10, 36, 37, 38, 39, 42, 44, 48, 49, 51, 58,
              59, 87, 138, 139, 140, 240, 246],

    "126_1": [229, 258, 170, 238, 296, 266, 48, 61, 254, 15, 290, 59, 172, 57,
              88, 113, 77, 268, 246, 235, 305, 160, 104, 218, 297, 187, 120,
              280, 262, 51, 56, 279, 41, 173, 45, 298, 265, 105, 84, 101, 267,
              278, 112, 206, 64, 94, 285, 12, 68, 301, 22, 133, 70, 152, 36,
              252, 275, 149, 272, 294, 286, 241, 153, 273, 200, 190, 231, 287,
              189, 93, 146, 23, 20, 293, 44, 184, 62, 185, 129, 28, 97, 119,
              210, 127, 53, 199, 26, 289, 302, 10, 25, 257, 216, 242, 80, 107,
              230, 32, 122, 264, 193, 208, 130, 151, 21, 128, 34, 244, 111,
              109, 211, 55, 168, 42, 192, 66, 217, 271, 118, 65, 39, 74, 14,
              239, 31, 19],

    "126_2": [86, 11, 267, 109, 278, 50, 264, 289, 41, 136, 126, 15, 70, 184,
              235, 199, 179, 98, 10, 202, 216, 218, 133, 304, 67, 271, 19, 256,
              79, 217, 122, 275, 124, 31, 34, 120, 112, 290, 24, 293, 52, 212,
              286, 138, 296, 258, 226, 69, 91, 297, 214, 40, 153, 94, 114, 175,
              58, 160, 168, 215, 84, 56, 77, 111, 223, 238, 63, 147, 113, 64,
              14, 49, 48, 140, 195, 68, 206, 61, 213, 62, 269, 229, 240, 46,
              299, 80, 78, 21, 25, 273, 220, 51, 252, 65, 208, 12, 200, 93, 87,
              210, 20, 33, 268, 139, 96, 105, 74, 174, 72, 243, 270, 295, 88,
              132, 219, 211, 16, 279, 53, 148, 60, 115, 176, 18, 73, 100],

    "126_3": [132, 74, 264, 271, 160, 42, 286, 92, 28, 269, 259, 267, 278, 111,
              215, 73, 181, 174, 49, 204, 220, 66, 12, 17, 190, 293, 156, 262,
              207, 99, 136, 94, 125, 153, 246, 279, 287, 161, 148, 234, 180,
              43, 106, 260, 189, 200, 226, 230, 14, 146, 152, 302, 47, 277,
              147, 281, 119, 22, 62, 19, 60, 67, 48, 109, 35, 112, 210, 256,
              72, 218, 16, 36, 64, 129, 202, 219, 253, 229, 266, 244, 33, 53,
              294, 27, 285, 83, 57, 187, 103, 176, 139, 296, 206, 183, 85, 32,
              96, 257, 118, 184, 282, 238, 55, 175, 108, 15, 77, 188, 46, 248,
              235, 169, 179, 275, 216, 38, 177, 39, 172, 151, 98, 252, 258,
              168, 284, 289],

    "126_4": [77, 117, 139, 82, 273, 96, 152, 110, 219, 213, 114, 68, 188, 102,
              78, 32, 258, 129, 99, 86, 216, 38, 113, 207, 246, 27, 71, 195,
              17, 122, 294, 171, 223, 278, 211, 254, 277, 59, 73, 21, 115, 138,
              244, 133, 36, 84, 58, 28, 11, 180, 226, 112, 123, 151, 88, 269,
              23, 127, 81, 160, 106, 52, 193, 289, 282, 176, 47, 299, 266, 264,
              168, 53, 231, 229, 296, 256, 97, 179, 126, 242, 80, 217, 279, 34,
              297, 159, 267, 268, 10, 224, 45, 55, 265, 109, 293, 65, 202, 262,
              164, 92, 305, 271, 252, 173, 25, 120, 260, 298, 287, 46, 69, 155,
              303, 31, 174, 290, 154, 234, 237, 74, 149, 57, 128, 62, 181,
              169],

    "126_5": [183, 89, 173, 85, 16, 13, 202, 270, 120, 169, 15, 276, 127, 124,
              140, 61, 188, 164, 244, 111, 231, 10, 64, 68, 298, 172, 283, 109,
              246, 146, 171, 36, 267, 60, 190, 154, 210, 67, 156, 223, 49, 37,
              52, 208, 206, 295, 43, 83, 287, 212, 195, 130, 98, 277, 65, 57,
              22, 101, 26, 263, 152, 248, 80, 291, 39, 280, 126, 161, 239, 105,
              112, 296, 56, 62, 14, 81, 304, 35, 129, 100, 92, 285, 168, 72,
              32, 84, 20, 265, 63, 31, 106, 179, 115, 34, 21, 28, 79, 235, 230,
              18, 260, 139, 305, 102, 82, 290, 200, 138, 38, 242, 297, 122, 87,
              301, 207, 180, 90, 104, 147, 176, 229, 204, 181, 219, 160, 119],

    "252": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
            45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 78, 79, 80,
            81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98,
            99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 117, 118, 119, 120, 122, 123, 124, 125, 126,
            127, 128, 129, 130, 132, 133, 136, 138, 139, 140, 146, 147, 148,
            149, 151, 152, 153, 154, 155, 156, 159, 160, 161, 164, 168, 169,
            170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 183, 184,
            185, 187, 188, 189, 190, 192, 193, 195, 198, 199, 200, 201, 202,
            204, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218,
            219, 220, 223, 224, 226, 229, 230, 231, 234, 235, 237, 238, 239,
            240, 241, 242, 243, 244, 246, 248, 252, 253, 254, 256, 257, 258,
            259, 260, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
            273, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286,
            287, 289, 290, 291, 293, 294, 295, 296, 297, 298, 299, 301, 302,
            303, 304, 305],
}
