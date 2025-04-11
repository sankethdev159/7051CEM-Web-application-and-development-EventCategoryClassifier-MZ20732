import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import java.util.ArrayList;
import java.util.Arrays;

public class EventClassifier {

    public static void main(String[] args) throws Exception {

        ArrayList<Attribute> featureList = new ArrayList<>();
        featureList.add(new Attribute("event_title", (ArrayList<String>) null));
        featureList.add(new Attribute("venue", (ArrayList<String>) null));
        featureList.add(new Attribute("date_str", (ArrayList<String>) null));
        featureList.add(new Attribute("details", (ArrayList<String>) null));
        ArrayList<String> categoryValues = new ArrayList<>(Arrays.asList("Conference", "Wedding", "Workshop", "Party"));
        featureList.add(new Attribute("category", categoryValues));

        Instances trainingSet = new Instances("EventTrainingData", featureList, 0);
        trainingSet.setClassIndex(4);

        loadTrainingData(trainingSet);

        StringToWordVector textFilter = new StringToWordVector();
        textFilter.setInputFormat(trainingSet);
        textFilter.setWordsToKeep(1000);
        textFilter.setDoNotOperateOnPerClassBasis(true);
        textFilter.setLowerCaseTokens(true);
        Instances vectorizedTrain = Filter.useFilter(trainingSet, textFilter);

        System.out.println("\n>>> Training complete. Classifier is ready!");

        SMO svmModel = new SMO();
        svmModel.buildClassifier(vectorizedTrain);

        Instances testSet = new Instances("EventTestData", featureList, 0);
        testSet.setClassIndex(4);

        addRecord(testSet, "Innovative Investment Forum", "London", "2025-05-15",
                "A forum for startup founders to pitch innovative funding ideas", "?");
        addRecord(testSet, "Starry Night Wedding Gala", "Los Angeles", "2025-08-22",
                "An opulent wedding celebration under a starlit sky", "?");
        addRecord(testSet, "Robotics & Automation Workshop", "Tokyo", "2025-04-18",
                "Hands-on session covering the latest in robotics technology", "?");
        addRecord(testSet, "Urban Music Carnival", "Berlin", "2025-03-30",
                "A vibrant carnival featuring live music and street art", "?");
        addRecord(testSet, "Gourmet Culinary Symposium", "Paris", "2025-06-05",
                "A symposium exploring modern gourmet cooking techniques", "?");
        addRecord(testSet, "Tech Startup Bootcamp", "San Francisco", "2025-07-12",
                "An intensive bootcamp for aspiring tech entrepreneurs", "?");
        addRecord(testSet, "Historical Banquet Evening", "Edinburgh", "2025-09-10",
                "A formal banquet set in a historic castle", "?");
        addRecord(testSet, "Outdoor Survival Skills Workshop", "Vancouver", "2025-05-22",
                "A workshop focused on wilderness survival and adventure", "?");
        addRecord(testSet, "Virtual Reality Showcase", "Singapore", "2025-10-03",
                "An expo featuring the latest virtual reality innovations", "?");
        addRecord(testSet, "Cultural Dance Extravaganza", "Mumbai", "2025-11-25",
                "A festival celebrating traditional and modern dance forms", "?");

        Instances vectorizedTest = Filter.useFilter(testSet, textFilter);
        System.out.println("\n>>> Event Predictions:\n");

        for (int i = 0; i < vectorizedTest.numInstances(); i++) {
            Instance currentInst = vectorizedTest.instance(i);
            double predIndex = svmModel.classifyInstance(currentInst);
            String predictedCategory = testSet.classAttribute().value((int) predIndex);
            System.out.println("Event: " + testSet.instance(i).stringValue(0)
                    + " | Venue: " + testSet.instance(i).stringValue(1)
                    + " | Predicted Category: " + predictedCategory);
        }
    }


    public static void addRecord(Instances data, String title, String venue, String date,
                                 String details, String category) {
        Instance newInst = new DenseInstance(5);
        newInst.setDataset(data);
        newInst.setValue(0, title);
        newInst.setValue(1, venue);
        newInst.setValue(2, date);
        newInst.setValue(3, details);
        if (!category.equals("?")) {
            newInst.setValue(4, data.classAttribute().indexOfValue(category));
        }
        data.add(newInst);
    }


    public static void loadTrainingData(Instances data) {
        // Conferences (13 events)
        addOriginalRecord(data, "International AI Summit", "London", "2025-02-20",
                "A premier conference discussing AI breakthroughs and ethical considerations", "Conference");
        addOriginalRecord(data, "Global Tech Forum", "New York", "2025-03-05",
                "Discussing global trends in technology and innovation", "Conference");
        addOriginalRecord(data, "Healthcare Innovations Conference", "Boston", "2025-04-12",
                "Conference on the future of healthcare technology", "Conference");
        addOriginalRecord(data, "Renewable Energy Summit", "Berlin", "2025-05-08",
                "Focus on renewable energy solutions and sustainability", "Conference");
        addOriginalRecord(data, "Blockchain Expo", "San Francisco", "2025-06-15",
                "Exploring blockchain technology applications across industries", "Conference");
        addOriginalRecord(data, "Cybersecurity World", "Singapore", "2025-07-10",
                "Discussing cybersecurity challenges and defense strategies", "Conference");
        addOriginalRecord(data, "Digital Transformation Conference", "Tokyo", "2025-08-18",
                "Examining the impact of digital transformation in business", "Conference");
        addOriginalRecord(data, "Smart Cities Forum", "Amsterdam", "2025-09-22",
                "Exploring innovative technologies for smarter urban living", "Conference");
        addOriginalRecord(data, "FinTech Revolution Summit", "Zurich", "2025-10-05",
                "Conference on fintech innovations and financial technology trends", "Conference");
        addOriginalRecord(data, "E-commerce Leaders Conference", "Los Angeles", "2025-11-12",
                "Bringing together e-commerce experts and industry leaders", "Conference");
        addOriginalRecord(data, "Biotech Innovations Symposium", "San Diego", "2025-12-03",
                "Exploring emerging biotechnology and life sciences", "Conference");
        addOriginalRecord(data, "Future of Transportation Summit", "Paris", "2025-03-20",
                "Innovations in transportation and mobility", "Conference");
        addOriginalRecord(data, "Sustainable Business Forum", "Sydney", "2025-04-25",
                "Discussing sustainable business practices and corporate responsibility", "Conference");

        // Weddings (12 events)
        addOriginalRecord(data, "Anna & Mark's Intimate Wedding", "Venice", "2025-06-15",
                "A romantic wedding held in a historic Venetian villa", "Wedding");
        addOriginalRecord(data, "Laura & David's Beach Wedding", "Miami", "2025-07-20",
                "A relaxed beachfront ceremony with tropical vibes", "Wedding");
        addOriginalRecord(data, "Sophia & Michael's Garden Wedding", "Charleston", "2025-05-30",
                "An elegant garden wedding with beautiful floral arrangements", "Wedding");
        addOriginalRecord(data, "Isabella & James's Rustic Wedding", "Nashville", "2025-08-12",
                "A charming rustic wedding with country-style decor", "Wedding");
        addOriginalRecord(data, "Emma & Liam's Classic Wedding", "New Orleans", "2025-09-05",
                "A timeless wedding featuring live jazz and elegant dining", "Wedding");
        addOriginalRecord(data, "Olivia & Noah's Vineyard Wedding", "Napa Valley", "2025-10-11",
                "A sophisticated wedding set among scenic vineyards", "Wedding");
        addOriginalRecord(data, "Mia & Ethan's Urban Wedding", "Chicago", "2025-11-03",
                "A modern wedding in a chic urban setting", "Wedding");
        addOriginalRecord(data, "Ava & Lucas's Destination Wedding", "Santorini", "2025-05-20",
                "A breathtaking destination wedding with stunning sea views", "Wedding");
        addOriginalRecord(data, "Charlotte & Benjamin's Country Wedding", "Texas", "2025-06-28",
                "A laid-back country wedding with Southern charm", "Wedding");
        addOriginalRecord(data, "Amelia & Oliver's Elegant Wedding", "Vienna", "2025-07-15",
                "An elegant wedding infused with European sophistication", "Wedding");
        addOriginalRecord(data, "Harper & William's Intimate Wedding", "Barcelona", "2025-08-22",
                "A small, intimate wedding in a vibrant city setting", "Wedding");
        addOriginalRecord(data, "Evelyn & Daniel's Modern Wedding", "Toronto", "2025-09-18",
                "A modern wedding blending traditional and contemporary styles", "Wedding");

        // Workshops (12 events)
        addOriginalRecord(data, "Full-Stack Web Development Workshop", "San Jose", "2025-01-15",
                "An intensive workshop covering modern web development frameworks", "Workshop");
        addOriginalRecord(data, "Data Science Bootcamp", "Chicago", "2025-02-25",
                "Hands-on training in data analysis and machine learning techniques", "Workshop");
        addOriginalRecord(data, "Photography Masterclass", "Paris", "2025-03-30",
                "Learn professional photography skills and techniques", "Workshop");
        addOriginalRecord(data, "Creative Writing Workshop", "Dublin", "2025-04-10",
                "Enhance your storytelling and creative writing skills", "Workshop");
        addOriginalRecord(data, "Digital Marketing Essentials", "London", "2025-05-05",
                "Workshop on effective digital marketing strategies and tools", "Workshop");
        addOriginalRecord(data, "Graphic Design Bootcamp", "Berlin", "2025-06-08",
                "Learn modern graphic design techniques in this intensive workshop", "Workshop");
        addOriginalRecord(data, "Mobile App Development Workshop", "San Francisco", "2025-07-12",
                "Build mobile applications using modern development tools", "Workshop");
        addOriginalRecord(data, "Public Speaking Workshop", "Toronto", "2025-08-20",
                "Improve your public speaking and presentation skills", "Workshop");
        addOriginalRecord(data, "Entrepreneurship 101", "New York", "2025-09-14",
                "Workshop for aspiring entrepreneurs covering startup essentials", "Workshop");
        addOriginalRecord(data, "Social Media Strategy Workshop", "Los Angeles", "2025-10-03",
                "Learn effective strategies for building your social media presence", "Workshop");
        addOriginalRecord(data, "Mindfulness and Meditation Workshop", "Bali", "2025-11-07",
                "Explore mindfulness practices and meditation techniques", "Workshop");
        addOriginalRecord(data, "Cooking Masterclass", "Rome", "2025-12-01",
                "Master the art of Italian cooking in this hands-on workshop", "Workshop");

        // Parties (13 events)
        addOriginalRecord(data, "Annual New Year's Eve Party", "New York", "2025-12-31",
                "Ring in the New Year with music, dancing, and fireworks", "Party");
        addOriginalRecord(data, "Summer Pool Party", "Miami", "2025-07-04",
                "Enjoy a refreshing pool party with great music and vibes", "Party");
        addOriginalRecord(data, "Halloween Costume Party", "Los Angeles", "2025-10-31",
                "A night of spooky fun and creative costumes", "Party");
        addOriginalRecord(data, "Spring Garden Party", "San Francisco", "2025-04-22",
                "Celebrate spring with an outdoor garden party", "Party");
        addOriginalRecord(data, "Independence Day Celebration", "Washington D.C.", "2025-07-04",
                "Festive celebration with parades, fireworks, and patriotic activities", "Party");
        addOriginalRecord(data, "Autumn Harvest Party", "Chicago", "2025-10-15",
                "Celebrate the season with local foods and live music", "Party");
        addOriginalRecord(data, "Charity Gala Party", "London", "2025-11-20",
                "A glamorous charity event with dinner and live entertainment", "Party");
        addOriginalRecord(data, "University Welcome Party", "Boston", "2025-09-01",
                "Kick off the academic year with a lively welcome party", "Party");
        addOriginalRecord(data, "Corporate Year-End Party", "Singapore", "2025-12-15",
                "Celebrate the year's achievements at this corporate event", "Party");
        addOriginalRecord(data, "Music Festival Party", "Austin", "2025-08-05",
                "Join a vibrant music festival featuring multiple live acts", "Party");
        addOriginalRecord(data, "Art & Culture Soirée", "Barcelona", "2025-05-18",
                "An elegant evening celebrating art and cultural diversity", "Party");
        addOriginalRecord(data, "Beach Bonfire Party", "Sydney", "2025-06-21",
                "Enjoy an evening of bonfire and beachside festivities", "Party");
        addOriginalRecord(data, "Midnight Dance Party", "Las Vegas", "2025-12-28",
                "Dance the night away at this high-energy celebration", "Party");
    }


    public static void addOriginalRecord(Instances data, String eventName, String venue,
                                         String date, String details, String category) {
        Instance record = new DenseInstance(5);
        record.setDataset(data);
        record.setValue(0, eventName);
        record.setValue(1, venue);
        record.setValue(2, date);
        record.setValue(3, details);
        if (!category.equals("?")) {
            record.setValue(4, data.classAttribute().indexOfValue(category));
        }
        data.add(record);
    }
}
