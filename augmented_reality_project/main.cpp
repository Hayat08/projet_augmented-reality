#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>

using namespace cv;
using namespace std;


// ======================================================
// 1. PARAMETRES
// ======================================================

string dossierRef = "Reference Images-20261114/";
string dossierVideo = "Overlay Videos-20261114/";

string nomsRef[] = {
    "1-Toyota", "2-BMW", "3-Honda", "4-Suzuki",
    "5-Peugeot", "6-Volkswagen", "7-Ford", "8-Hyundai",
    "9-Mercedes", "10-Citroen", "11-Range-Rover", "12-Nissan"
};

// Citroen et Range-Rover : numerotation video inversee
string nomsVideo[] = {
    "1-Toyota", "2-BMW", "3-Honda", "4-Suzuki",
    "5-Peugeot", "6-Volkswagen", "7-Ford", "8-Hyundai",
    "9-Mercedes", "11-Citroen", "10-Range-Rover", "12-Nissan"
};

int largeurTraitement = 640;     // largeur de la frame webcam traitee
int tailleMaxRef = 480;     // les references sont reduites (plus rapide)
int largeurVideoIncr = 320;     // les videos sont reduites avant incrustation

int seuilAppariements = 10;      // correspondances minimum avant RANSAC
int seuilVraisPositifs = 12;     // inliers RANSAC minimum pour une detection
int seuilInliersSuivi = 8;      // inliers minimum pour continuer le suivi
double seuilFlot = 40;           // deplacement max d'un coin entre 2 images

int periodeDetection = 2;        // detection toutes les N images
int maxDetectionsParImage = 3;   // nb max de references testees par image (round-robin)

bool afficherContours = true;


// ======================================================
// 2. STRUCTURES
// ======================================================

struct Cible
{
    string nom;

    Mat image;                       // reference (redimensionnee)
    vector<KeyPoint> kp;
    Mat desc;

    bool visible = false;
    bool tracking = false;
    string mode;
    int nbAppariements = 0;

    vector<Point2f> sommets;
    vector<Point2f> prevSommets;
    Mat H;

    // Points suivis par flot optique (coordonnees ref <-> scene)
    vector<Point2f> ptsRef;
    vector<Point2f> ptsScene;

    // Video de recalage
    Ptr<VideoCapture> video;
    Mat frameVideo;
    Mat masqueBlanc;
    double fpsVideo = 25;
    double dernierMs = 0;
};

struct ResultatDetection
{
    int id;
    bool detected;
    vector<Point2f> corners;
    Mat homography;
};

vector<Cible> cibles;

Ptr<Feature2D> detecteur;
Ptr<BFMatcher> matcher;
string nomAlgo;

VideoCapture cap;

vector<Mat> prevPyr, currPyr;
bool prevPyrValide = false;
const Size winLK(21, 21);
const int niveauxLK = 3;

int imageCourante = 0;
int curseurDetection = 0;


// ======================================================
// 3. OUTILS
// ======================================================

vector<Point> versPoints(const vector<Point2f>& pts)
{
    vector<Point> res;
    res.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); i++)
        res.push_back(Point((int)pts[i].x, (int)pts[i].y));
    return res;
}

void remplirZone(Mat& masque, const vector<Point2f>& sommets)
{
    vector<vector<Point> > polys;
    polys.push_back(versPoints(sommets));
    fillPoly(masque, polys, Scalar(255));
}

double tempsEcoule(double t0)
{
    return ((double)getTickCount() - t0) / getTickFrequency() * 1000;
}

// Ajoute un separateur final si le dossier n'en a pas
string avecSeparateur(const string& dossier)
{
    if (dossier.empty())
        return dossier;

    char dernier = dossier[dossier.size() - 1];

    if (dernier == '/' || dernier == '\\')
        return dossier;

    return dossier + "/";
}

string trouverFichier(const string& dossier, const string& base,
    const vector<string>& extensions)
{
    for (size_t i = 0; i < extensions.size(); i++)
    {
        string chemin = avecSeparateur(dossier) + base + extensions[i];
        ifstream f(chemin.c_str());
        if (f.good())
            return chemin;
    }
    return "";
}

// Recadrage centre au ratio voulu, puis reduction
Mat preparerVideo(const Mat& v, double ratio, int largeurCible)
{
    double r = (double)v.cols / v.rows;
    Rect roi(0, 0, v.cols, v.rows);

    if (r > ratio)
    {
        int w = (int)(v.rows * ratio);
        roi = Rect((v.cols - w) / 2, 0, w, v.rows);
    }
    else
    {
        int h = (int)(v.cols / ratio);
        roi = Rect(0, (v.rows - h) / 2, v.cols, h);
    }

    Mat res;
    Size s(largeurCible, max(1, (int)(largeurCible / ratio)));
    resize(v(roi), res, s, 0, 0, INTER_AREA);
    return res;
}


// ======================================================
// 4. CHOIX DE L'ALGORITHME
// ======================================================

bool choisirDetecteur()
{
    int choix;

    cout << "\n================================\n";
    cout << " RECONNAISSANCE MULTI-IMAGES\n";
    cout << "================================\n";
    cout << "1 - ORB\n2 - AKAZE\nVotre choix : ";
    cin >> choix;

    if (choix == 1)
    {
        detecteur = ORB::create(1000);
        nomAlgo = "ORB";
    }
    else if (choix == 2)
    {
        detecteur = AKAZE::create();
        nomAlgo = "AKAZE";
    }
    else
    {
        cout << "Choix invalide.\n";
        return false;
    }

    // ORB et AKAZE : descripteurs binaires -> Hamming
    matcher = makePtr<BFMatcher>(NORM_HAMMING);

    cout << "Algorithme " << nomAlgo << " selectionne.\n";
    return true;
}


// ======================================================
// 5. CARACTERISTIQUES (sur image deja en niveaux de gris)
// ======================================================

void calculerCaracteristiques(const Mat& gris, InputArray masque,
    vector<KeyPoint>& kp, Mat& desc)
{
    detecteur->detectAndCompute(gris, masque, kp, desc);
}


// ======================================================
// 6. CHARGEMENT DES REFERENCES ET VIDEOS
// ======================================================

bool chargerReferences()
{
    int nbRef = sizeof(nomsRef) / sizeof(nomsRef[0]);

    for (int i = 0; i < nbRef; i++)
    {
        Cible c;
        c.nom = nomsRef[i];

        string cheminImg = trouverFichier(
            dossierRef, nomsRef[i], { ".png", ".jpg", ".jpeg", ".webp" });

        Mat img = cheminImg.empty() ? Mat() : imread(cheminImg);

        if (img.empty())
        {
            cerr << "Image introuvable : " << avecSeparateur(dossierRef) + nomsRef[i] << endl;
            return false;
        }

        // Reduire la reference si elle est grande
        int mx = max(img.cols, img.rows);
        if (mx > tailleMaxRef)
        {
            double s = (double)tailleMaxRef / mx;
            resize(img, img, Size(), s, s, INTER_AREA);
        }
        c.image = img;

        Mat gris;
        cvtColor(c.image, gris, COLOR_BGR2GRAY);
        calculerCaracteristiques(gris, noArray(), c.kp, c.desc);

        if (c.desc.empty() || (int)c.kp.size() < seuilAppariements)
        {
            cerr << "Pas assez de points dans : " << c.nom << endl;
            return false;
        }

        string cheminVid = trouverFichier(
            dossierVideo, nomsVideo[i], { ".mp4", ".avi", ".mov", ".mkv", ".wmv" });

        c.video = makePtr<VideoCapture>(cheminVid);

        if (cheminVid.empty() || !c.video->isOpened())
        {
            cerr << "Video introuvable ou illisible : "
                << avecSeparateur(dossierVideo) + nomsVideo[i] << endl;
            return false;
        }

        c.fpsVideo = max(1.0, c.video->get(CAP_PROP_FPS));
        c.dernierMs = 0;

        cibles.push_back(c);

        cout << "Reference chargee : " << c.nom
            << " | Points : " << c.kp.size()
            << " | Video : " << nomsVideo[i] << endl;
    }

    return true;
}


// ======================================================
// 7. APPARIEMENT (ratio de Lowe)
// ======================================================

int appariement(const Cible& c,
    const Mat& descScene,
    const vector<KeyPoint>& kpScene,
    vector<Point2f>& ptsRef,
    vector<Point2f>& ptsScene)
{
    ptsRef.clear();
    ptsScene.clear();

    if (descScene.empty() || c.desc.empty())
        return 0;

    vector<vector<DMatch> > knn;
    matcher->knnMatch(c.desc, descScene, knn, 2);

    for (size_t i = 0; i < knn.size(); i++)
    {
        if (knn[i].size() < 2)
            continue;

        if (knn[i][0].distance < 0.75f * knn[i][1].distance)
        {
            ptsRef.push_back(c.kp[knn[i][0].queryIdx].pt);
            ptsScene.push_back(kpScene[knn[i][0].trainIdx].pt);
        }
    }

    return (int)ptsRef.size();
}


// ======================================================
// 8. HOMOGRAPHIE
// ======================================================

int calculerHomographie(const vector<Point2f>& ptsRef,
    const vector<Point2f>& ptsScene,
    Mat& H,
    vector<uchar>& masque)
{
    if (ptsRef.size() < 4)
        return 0;

    H = findHomography(ptsRef, ptsScene, RANSAC, 3, masque);

    if (H.empty())
        return 0;

    return countNonZero(masque);
}

void garderInliers(vector<Point2f>& ref, vector<Point2f>& scene,
    const vector<uchar>& masque)
{
    vector<Point2f> r, s;
    for (size_t i = 0; i < masque.size(); i++)
    {
        if (masque[i])
        {
            r.push_back(ref[i]);
            s.push_back(scene[i]);
        }
    }
    ref.swap(r);
    scene.swap(s);
}


// ======================================================
// 9. SOMMETS + VALIDATION
// ======================================================

void calculerSommets(const Mat& H, const Cible& c, vector<Point2f>& sommets)
{
    vector<Point2f> coins(4);
    coins[0] = Point2f(0, 0);
    coins[1] = Point2f((float)c.image.cols, 0);
    coins[2] = Point2f((float)c.image.cols, (float)c.image.rows);
    coins[3] = Point2f(0, (float)c.image.rows);

    perspectiveTransform(coins, sommets, H);
}

bool sommetsValides(const vector<Point2f>& sommets, Size taille)
{
    if (sommets.size() != 4)
        return false;

    double aire = contourArea(sommets);

    if (aire < 400 || aire > 0.9 * taille.width * taille.height)
        return false;

    if (!isContourConvex(versPoints(sommets)))
        return false;

    for (size_t i = 0; i < sommets.size(); i++)
    {
        if (sommets[i].x < -100 || sommets[i].x > taille.width + 100 ||
            sommets[i].y < -100 || sommets[i].y > taille.height + 100)
            return false;
    }

    return true;
}


// ======================================================
// 10. TRACKING : flot optique sur les points interieurs
//     + re-estimation de l'homographie (plus robuste que 4 coins)
// ======================================================

bool suivreCible(Cible& c, Size taille)
{
    if (c.ptsScene.size() < (size_t)seuilInliersSuivi)
        return false;

    vector<Point2f> nouveaux;
    vector<uchar> status;
    vector<float> erreurs;

    calcOpticalFlowPyrLK(prevPyr, currPyr, c.ptsScene, nouveaux,
        status, erreurs, winLK, niveauxLK);

    vector<Point2f> ref, scene;
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i] && erreurs[i] < 30)
        {
            ref.push_back(c.ptsRef[i]);
            scene.push_back(nouveaux[i]);
        }
    }

    if ((int)ref.size() < seuilInliersSuivi)
        return false;

    Mat H;
    vector<uchar> masque;
    int nb = calculerHomographie(ref, scene, H, masque);

    if (nb < seuilInliersSuivi)
        return false;

    vector<Point2f> sommets;
    calculerSommets(H, c, sommets);

    if (!sommetsValides(sommets, taille))
        return false;

    // Saut trop grand d'une image a l'autre -> suivi perdu
    for (size_t i = 0; i < 4; i++)
        if (norm(sommets[i] - c.prevSommets[i]) > seuilFlot)
            return false;

    garderInliers(ref, scene, masque);

    c.ptsRef = ref;
    c.ptsScene = scene;
    c.H = H;
    c.sommets = sommets;
    c.prevSommets = sommets;

    return true;
}


// ======================================================
// 11. TRAITEMENT (tracking + detection)
// ======================================================

void traiterImage(const Mat& gris, double& tSuivi, double& tDetection)
{
    // Pyramide de l'image courante : calculee une seule fois pour toutes les cibles
    buildOpticalFlowPyramid(gris, currPyr, winLK, niveauxLK);

    Mat occupe = Mat::zeros(gris.size(), CV_8UC1);

    // --- A. TRACKING ---
    double t0 = (double)getTickCount();

    for (size_t i = 0; i < cibles.size(); i++)
    {
        Cible& c = cibles[i];
        c.visible = false;

        if (!c.tracking)
            continue;

        if (prevPyrValide && suivreCible(c, gris.size()))
        {
            c.visible = true;
            c.mode = "Tracking";
            remplirZone(occupe, c.sommets);
        }
        else
        {
            c.tracking = false;
        }
    }

    tSuivi = tempsEcoule(t0);

    // --- B. DETECTION (round-robin, quelques cibles par image) ---
    t0 = (double)getTickCount();

    if (imageCourante % periodeDetection == 0)
    {
        int n = (int)cibles.size();

        // Y a-t-il quelque chose a detecter ?
        bool besoin = false;
        for (int i = 0; i < n; i++)
            if (!cibles[i].tracking) { besoin = true; break; }

        if (besoin)
        {
            // Caracteristiques de la scene : une seule fois, hors zones deja occupees
            Mat libre;
            bitwise_not(occupe, libre);

            vector<KeyPoint> kpScene;
            Mat descScene;
            calculerCaracteristiques(gris, libre, kpScene, descScene);

            int tentatives = 0;
            int dernier = curseurDetection;

            for (int k = 0; k < n && tentatives < maxDetectionsParImage; k++)
            {
                int i = (curseurDetection + k) % n;
                Cible& c = cibles[i];

                if (c.tracking)
                    continue;

                tentatives++;
                dernier = (i + 1) % n;

                vector<Point2f> ptsRef, ptsScene;
                c.nbAppariements = appariement(c, descScene, kpScene, ptsRef, ptsScene);

                if (c.nbAppariements < seuilAppariements)
                    continue;

                Mat H;
                vector<uchar> masque;
                int nbInliers = calculerHomographie(ptsRef, ptsScene, H, masque);

                if (nbInliers < seuilVraisPositifs)
                    continue;

                vector<Point2f> sommets;
                calculerSommets(H, c, sommets);

                if (!sommetsValides(sommets, gris.size()))
                    continue;

                garderInliers(ptsRef, ptsScene, masque);

                c.ptsRef = ptsRef;
                c.ptsScene = ptsScene;
                c.H = H;
                c.sommets = sommets;
                c.prevSommets = sommets;

                c.visible = true;
                c.tracking = true;
                c.mode = "Detection";

                remplirZone(occupe, sommets);
            }

            curseurDetection = dernier;
        }
    }

    tDetection = tempsEcoule(t0);
}


// ======================================================
// 12. AUGMENTATION : incrustation video (limitee a la zone utile)
// ======================================================

bool lireVideo(Cible& c, Mat& f)
{
    if (!c.video->read(f))
    {
        c.video->set(CAP_PROP_POS_FRAMES, 0);
        if (!c.video->read(f))
            return false;
    }
    return !f.empty();
}

void lireImageVideo(Cible& c, double maintenantMs)
{
    if (c.video.empty() || !c.video->isOpened())
        return;

    double periode = 1000.0 / c.fpsVideo;
    double dt = maintenantMs - c.dernierMs;
    int n = (int)(dt / periode);

    if (!c.frameVideo.empty() && n < 1)
        return;

    if (c.frameVideo.empty())
        n = 1;

    // Si la cible etait cachee longtemps, on ne rattrape pas tout
    bool trop = n > 3;
    if (trop) n = 1;

    Mat f;
    // On saute les images en retard (grab = decodage sans conversion)
    for (int k = 0; k < n - 1; k++)
    {
        if (!c.video->grab())
        {
            c.video->set(CAP_PROP_POS_FRAMES, 0);
            c.video->grab();
        }
    }

    if (lireVideo(c, f))
    {
        c.frameVideo = preparerVideo(
            f, (double)c.image.cols / c.image.rows, largeurVideoIncr);

        if (c.masqueBlanc.size() != c.frameVideo.size())
            c.masqueBlanc = Mat(c.frameVideo.size(), CV_8UC1, Scalar(255));
    }

    c.dernierMs = trop ? maintenantMs : c.dernierMs + n * periode;
}

void augmenter(Mat& img, Cible& c, double maintenantMs)
{
    lireImageVideo(c, maintenantMs);

    if (c.frameVideo.empty())
        return;

    // Zone englobante clippee : on ne deforme que ce qui est utile
    Rect roi = boundingRect(c.sommets) & Rect(0, 0, img.cols, img.rows);
    if (roi.width < 2 || roi.height < 2)
        return;

    vector<Point2f> coinsVideo(4);
    coinsVideo[0] = Point2f(0, 0);
    coinsVideo[1] = Point2f((float)c.frameVideo.cols, 0);
    coinsVideo[2] = Point2f((float)c.frameVideo.cols, (float)c.frameVideo.rows);
    coinsVideo[3] = Point2f(0, (float)c.frameVideo.rows);

    Mat M = getPerspectiveTransform(coinsVideo, c.sommets);

    // Decalage vers le repere du ROI
    Mat T = (Mat_<double>(3, 3) << 1, 0, -roi.x, 0, 1, -roi.y, 0, 0, 1);
    Mat Mr = T * M;

    Mat videoDeformee, masque;
    warpPerspective(c.frameVideo, videoDeformee, Mr, roi.size(), INTER_LINEAR);
    warpPerspective(c.masqueBlanc, masque, Mr, roi.size(), INTER_NEAREST);

    Mat zone = img(roi);
    videoDeformee.copyTo(zone, masque);
}


// ======================================================
// 13. AFFICHAGE
// ======================================================

void dessinerCible(Mat& img, const Cible& c)
{
    vector<Point> contour = versPoints(c.sommets);

    polylines(img, contour, true, Scalar(0, 255, 0), 2);

    putText(img, c.nom + " : " + c.mode,
        Point(contour[0].x, max(contour[0].y - 8, 12)),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
}


// ======================================================
// 14. RESULTATS POUR LE BINOME (partie recalage)
// ======================================================

vector<ResultatDetection> recupererResultats()
{
    vector<ResultatDetection> resultats;

    for (size_t i = 0; i < cibles.size(); i++)
    {
        ResultatDetection r;
        r.id = (int)i;
        r.detected = cibles[i].visible;
        r.corners = cibles[i].sommets;
        r.homography = cibles[i].H;
        resultats.push_back(r);
    }

    return resultats;
}


// ======================================================
// 15. MAIN
// ======================================================

int main()
{
    if (!choisirDetecteur())
        return -1;

    if (!chargerReferences())
        return -1;

    cout << "\nNombre de references chargees : " << cibles.size() << endl;

    cap.open(0);

    if (!cap.isOpened())
    {
        cerr << "Impossible d'ouvrir la webcam." << endl;
        return -1;
    }

    // Demander directement une petite resolution a la camera
    cap.set(CAP_PROP_FRAME_WIDTH, largeurTraitement);
    cap.set(CAP_PROP_FRAME_HEIGHT, largeurTraitement * 3 / 4);
    cap.set(CAP_PROP_BUFFERSIZE, 1);

    cout << "Webcam ouverte. ECHAP pour quitter." << endl;

    ofstream log("perf_" + nomAlgo + ".csv");
    log << "image;suivi_ms;detection_ms;augmentation_ms;total_ms;fps;nb_visibles\n";

    Mat frame, gris;

    double tDebut = (double)getTickCount();
    double tBoucle = 0;

    while (true)
    {
        cap >> frame;

        if (frame.empty())
            break;

        double t0Boucle = (double)getTickCount();

        if (largeurTraitement > 0 && frame.cols > largeurTraitement)
        {
            resize(frame, frame,
                Size(largeurTraitement,
                    frame.rows * largeurTraitement / frame.cols),
                0, 0, INTER_AREA);
        }

        cvtColor(frame, gris, COLOR_BGR2GRAY);

        double tSuivi = 0, tDetection = 0;

        traiterImage(gris, tSuivi, tDetection);

        vector<ResultatDetection> resultats = recupererResultats();

        // Augmentation
        double tAug0 = (double)getTickCount();
        double maintenant = tempsEcoule(tDebut);
        int nbVisibles = 0;

        for (size_t i = 0; i < cibles.size(); i++)
        {
            if (cibles[i].visible)
            {
                augmenter(frame, cibles[i], maintenant);

                if (afficherContours)
                    dessinerCible(frame, cibles[i]);

                nbVisibles++;
            }
        }

        double tAugmentation = tempsEcoule(tAug0);
        double tTotal = tempsEcoule(t0Boucle);

        tBoucle = (tBoucle == 0) ? tTotal : 0.9 * tBoucle + 0.1 * tTotal;
        double fps = 1000.0 / tBoucle;

        log << imageCourante << ";" << tSuivi << ";" << tDetection << ";"
            << tAugmentation << ";" << tTotal << ";"
            << 1000.0 / max(tTotal, 0.001) << ";" << nbVisibles << "\n";

        putText(frame,
            nomAlgo + " | Images detectees : " + to_string(nbVisibles) +
            " / " + to_string(cibles.size()),
            Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);

        putText(frame,
            "Tracking : " + to_string((int)tSuivi) + " ms | Detection : " +
            to_string((int)tDetection) + " ms | Aug. : " +
            to_string((int)tAugmentation) + " ms",
            Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 0, 0), 1);

        putText(frame, format("%.1f FPS (%.1f ms)", fps, tBoucle),
            Point(10, 75), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);

        imshow("Detection et Tracking Multi-Images", frame);

        // La pyramide courante devient la precedente (sans recalcul)
        prevPyr.swap(currPyr);
        prevPyrValide = true;

        imageCourante++;

        if (waitKey(1) == 27)
            break;
    }

    log.close();
    cap.release();
    destroyAllWindows();

    return 0;
}