
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
#include <sstream>
#include <algorithm>
#include <map>

using namespace cv;
using namespace std;


string dossierRef = "Reference Images-20261114/";



string nomsRef[] = {
    "1-Toyota.png",
    "2-BMW.jpg",
    "3-Honda.png"
};

int largeurTraitement = 640;

// Nombre minimum de correspondances
int seuilAppariements = 10;

// Nombre minimum de vrais positifs RANSAC
int seuilVraisPositifs = 8;

// Seuil de deplacement des coins pour le tracking
double seuilFlot = 40;

// Detection a chaque image
int periodeDetection = 1;


// ======================================================
// 2. STRUCTURES
// ======================================================

// Informations d'une image de reference

struct Cible
{
    string nom;

    Mat image;

    vector<KeyPoint> kp;
    Mat desc;

    bool visible;
    bool tracking;

    string mode;

    int nbAppariements;

    vector<Point2f> sommets;
    vector<Point2f> prevSommets;
};


// Resultat transmis a la partie recalage

struct ResultatDetection
{
    int id;
    bool detected;

    vector<Point2f> corners;
    Mat homography;
};


// Donnees globales

vector<Cible> cibles;

Ptr<Feature2D> detecteur;

VideoCapture cap;

Mat prevGris;

int imageCourante = 0;


// ======================================================
// 3. OUTILS
// ======================================================

// Convertir les points flottants en points entiers

vector<Point> versPoints(
    const vector<Point2f>& pts)
{
    vector<Point> res;

    for (size_t i = 0; i < pts.size(); i++)
    {
        res.push_back(
            Point(
                (int)pts[i].x,
                (int)pts[i].y
            )
        );
    }

    return res;
}


// Remplir une zone pour eviter les detections en double

void remplirZone(
    Mat& masque,
    const vector<Point2f>& sommets)
{
    vector<vector<Point> > polygones;

    polygones.push_back(versPoints(sommets));

    fillPoly(
        masque,
        polygones,
        Scalar(255)
    );
}


// Mesurer le temps d'execution

double tempsEcoule(double t0)
{
    return (
        (double)getTickCount() - t0
        ) / getTickFrequency() * 1000;
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

    cout << "1 - ORB\n";
    cout << "2 - AKAZE\n";

    cout << "Votre choix : ";
    cin >> choix;

    if (choix == 1)
    {
        detecteur = ORB::create(1500);

        cout << "Algorithme ORB selectionne.\n";
    }
    else if (choix == 2)
    {
        detecteur = AKAZE::create();

        cout << "Algorithme AKAZE selectionne.\n";
    }
    else
    {
        cout << "Choix invalide.\n";
        return false;
    }

    return true;
}


// ======================================================
// 5. CALCUL DES CARACTERISTIQUES
// ======================================================

void calculerCaracteristiques(
    const Mat& img,
    vector<KeyPoint>& kp,
    Mat& desc)
{
    Mat gris;

    cvtColor(
        img,
        gris,
        COLOR_BGR2GRAY
    );

    // Detection et calcul des descripteurs

    detecteur->detectAndCompute(
        gris,
        noArray(),
        kp,
        desc
    );
}


// ======================================================
// 6. CHARGEMENT DES IMAGES DE REFERENCE
// ======================================================

bool chargerReferences()
{
    int nbRef =
        sizeof(nomsRef) / sizeof(nomsRef[0]);

    for (int i = 0; i < nbRef; i++)
    {
        Cible c;

        c.nom = nomsRef[i];

        // Charger l'image

        c.image = imread(
            dossierRef + nomsRef[i]
        );

        if (c.image.empty())
        {
            cerr << "Image introuvable : "
                << nomsRef[i] << endl;

            return false;
        }

        // Calculer les points et descripteurs

        calculerCaracteristiques(
            c.image,
            c.kp,
            c.desc
        );

        if (c.desc.empty() ||
            c.kp.size() < seuilAppariements)
        {
            cerr << "Pas assez de points dans : "
                << c.nom << endl;

            return false;
        }

        c.visible = false;
        c.tracking = false;
        c.nbAppariements = 0;

        cibles.push_back(c);

        cout << "Reference chargee : "
            << c.nom
            << " | Points : "
            << c.kp.size()
            << endl;
    }

    return true;
}


// ======================================================
// 7. APPARIEMENT DES DESCRIPTEURS
// ======================================================

int appariement(
    const Cible& c,
    const Mat& descScene,
    const vector<KeyPoint>& kpScene,
    const Mat& occupe,
    vector<Point2f>& ptsRef,
    vector<Point2f>& ptsScene)
{
    ptsRef.clear();
    ptsScene.clear();

    if (descScene.empty() || c.desc.empty())
        return 0;

    // ORB et AKAZE par defaut utilisent
    // des descripteurs binaires.

    BFMatcher matcher(NORM_HAMMING);

    vector<vector<DMatch> > knn;

    matcher.knnMatch(
        c.desc,
        descScene,
        knn,
        2
    );

    // Filtrage des correspondances

    for (size_t i = 0; i < knn.size(); i++)
    {
        if (knn[i].size() < 2)
            continue;

        if (knn[i][0].distance <
            0.8 * knn[i][1].distance)
        {
            Point2f p =
                kpScene[knn[i][0].trainIdx].pt;

            int x = (int)p.x;
            int y = (int)p.y;

            if (x < 0 || x >= occupe.cols ||
                y < 0 || y >= occupe.rows)
                continue;

            // Ignorer les points deja occupes
            if (occupe.at<uchar>(y, x) == 0)
            {
                ptsRef.push_back(
                    c.kp[knn[i][0].queryIdx].pt
                );

                ptsScene.push_back(p);
            }
        }
    }

    return (int)ptsRef.size();
}


// ======================================================
// 8. CALCUL DE L'HOMOGRAPHIE
// ======================================================

int calculerHomographie(
    const vector<Point2f>& ptsRef,
    const vector<Point2f>& ptsScene,
    Mat& H)
{
    if (ptsRef.size() < 4)
        return 0;

    vector<uchar> masque;

    H = findHomography(
        ptsRef, //Points de l'image de référence
        ptsScene, //Points correspondants dans la webcam
        RANSAC,//Méthode utilisée pour calculer l'homographie
        3, // Seuil de distance en pixels pour accepter une correspondance comme inlier
        masque //Indique quelles correspondances sont considérées comme des inliers
    );

    if (H.empty())
        return 0;

    return countNonZero(masque);
}


// ======================================================
// 9. CALCUL DES QUATRE SOMMETS
// ======================================================

void calculerSommets(
    const Mat& H,
    const Cible& c,
    vector<Point2f>& sommets)
{
    vector<Point2f> coins(4);

    coins[0] = Point2f(0, 0);

    coins[1] = Point2f(
        (float)c.image.cols,
        0
    );

    coins[2] = Point2f(
        (float)c.image.cols,
        (float)c.image.rows
    );

    coins[3] = Point2f(
        0,
        (float)c.image.rows
    );

    perspectiveTransform(
        coins,
        sommets,
        H
    );
}


// ======================================================
// 10. VALIDATION DES SOMMETS
// ======================================================

bool sommetsValides(
    const vector<Point2f>& sommets,
    Size taille)
{
    if (sommets.size() != 4)
        return false;

    double aire = contourArea(sommets);

    if (aire < 400 ||
        aire > 0.9 * taille.width * taille.height)
    {
        return false;
    }

    // Verifier que le quadrilatere est convexe

    vector<Point> points = versPoints(sommets);

    if (!isContourConvex(points))
        return false;

    // Verifier que les sommets sont dans une zone raisonnable

    for (size_t i = 0; i < sommets.size(); i++)
    {
        if (sommets[i].x < -100 ||
            sommets[i].x > taille.width + 100 ||
            sommets[i].y < -100 ||
            sommets[i].y > taille.height + 100)
        {
            return false;
        }
    }

    return true;
}


// ======================================================
// 11. TRACKING PAR FLOT OPTIQUE
// ======================================================

bool suivreSommets(
    const Mat& precedente,
    const Mat& courante,
    const vector<Point2f>& prevSommets,
    vector<Point2f>& sommets)
{
    if (prevSommets.size() != 4)
        return false;

    vector<uchar> status;
    vector<float> erreurs;

    calcOpticalFlowPyrLK(
        precedente,
        courante,
        prevSommets,
        sommets,
        status,
        erreurs
    );

    double erreurMax = 0;

    for (size_t i = 0; i < sommets.size(); i++)
    {
        if (!status[i])
            return false;

        erreurMax = max(
            erreurMax,
            norm(sommets[i] - prevSommets[i])
        );
    }

    if (erreurMax > seuilFlot)
        return false;

    return true;
}


// ======================================================
// 12. TRAITEMENT DE L'IMAGE
// ======================================================

void traiterImage(
    const Mat& frame,
    const Mat& gris,
    double& tSuivi,
    double& tDetection)
{
    // Masque des zones deja occupees

    Mat occupe = Mat::zeros(
        gris.size(),
        CV_8UC1
    );

    // --------------------------------------
    // A. TRACKING DES IMAGES DEJA DETECTEES
    // --------------------------------------

    double t0 = (double)getTickCount();

    for (size_t i = 0; i < cibles.size(); i++)
    {
        Cible& c = cibles[i];

        c.visible = false;

        if (c.tracking && !prevGris.empty())
        {
            vector<Point2f> nouveauxSommets;

            if (suivreSommets(
                prevGris,
                gris,
                c.prevSommets,
                nouveauxSommets))
            {
                if (sommetsValides(
                    nouveauxSommets,
                    gris.size()))
                {
                    c.sommets = nouveauxSommets;

                    c.prevSommets = nouveauxSommets;

                    c.visible = true;

                    c.mode = "Tracking";

                    remplirZone(
                        occupe,
                        c.sommets
                    );
                }
                else
                {
                    c.tracking = false;
                }
            }
            else
            {
                c.tracking = false;
            }
        }
    }

    tSuivi = tempsEcoule(t0);


    // --------------------------------------
    // B. DETECTION DES IMAGES NON SUIVIES
    // --------------------------------------

    t0 = (double)getTickCount();

    if (imageCourante % periodeDetection == 0)
    {
        vector<KeyPoint> kpScene;

        Mat descScene;

        bool calcule = false;

        for (size_t i = 0; i < cibles.size(); i++)
        {
            Cible& c = cibles[i];

            // Si deja suivie, passer a la suivante

            if (c.tracking)
                continue;

            // Calculer les caracteristiques de la scene
            // une seule fois par frame

            if (!calcule)
            {
                calculerCaracteristiques(
                    frame,
                    kpScene,
                    descScene
                );

                calcule = true;
            }

            // Apparier les descripteurs

            vector<Point2f> ptsRef;
            vector<Point2f> ptsScene;

            c.nbAppariements = appariement(
                c,
                descScene,
                kpScene,
                occupe,
                ptsRef,
                ptsScene
            );

            // Calculer l'homographie

            Mat H;

            int nbInliers = calculerHomographie(
                ptsRef,
                ptsScene,
                H
            );

            if (nbInliers >= seuilVraisPositifs)
            {
                vector<Point2f> sommets;

                calculerSommets(
                    H,
                    c,
                    sommets
                );

                if (sommetsValides(
                    sommets,
                    gris.size()))
                {
                    c.sommets = sommets;

                    c.prevSommets = sommets;

                    c.visible = true;

                    c.tracking = true;

                    c.mode = "Detection";

                    remplirZone(
                        occupe,
                        sommets
                    );
                }
            }
        }
    }

    tDetection = tempsEcoule(t0);
}


// ======================================================
// 13. AFFICHAGE DES RESULTATS
// ======================================================

void dessinerCible(
    Mat& img,
    const Cible& c)
{
    vector<Point> contour =
        versPoints(c.sommets);

    // Contour de l'image reconnue

    polylines(
        img,
        contour,
        true,
        Scalar(0, 255, 0),
        2
    );

    // Afficher les quatre coins

    for (size_t i = 0; i < contour.size(); i++)
    {
        circle(
            img,
            contour[i],
            5,
            Scalar(0, 0, 255),
            FILLED
        );
    }

    // Nom de la reference et mode

    putText(
        img,
        c.nom + " : " + c.mode,
        Point(
            contour[0].x,
            max(contour[0].y - 8, 12)
        ),
        FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar(0, 255, 255),
        1
    );
}


// ======================================================
// 14. CREATION DES RESULTATS POUR LE BINOME
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

        resultats.push_back(r);
    }

    return resultats;
}


// ======================================================
// 15. MAIN
// ======================================================

int main()
{
    // Choisir ORB ou AKAZE

    if (!choisirDetecteur())
        return -1;


    // Charger les images de reference

    if (!chargerReferences())
        return -1;

    cout << "\nNombre de references chargees : "
        << cibles.size() << endl;


    // Ouvrir la webcam

    cap.open(0);

    if (!cap.isOpened())
    {
        cerr << "Impossible d'ouvrir la webcam." << endl;
        return -1;
    }

    cout << "Webcam ouverte avec succes." << endl;
    cout << "ECHAP pour quitter." << endl;


    // Boucle principale

    Mat frame;
    Mat gris;

    while (true)
    {
        cap >> frame;

        if (frame.empty())
            break;

        // Redimensionner la frame

        if (largeurTraitement > 0 &&
            frame.cols > largeurTraitement)
        {
            resize(
                frame,
                frame,
                Size(
                    largeurTraitement,
                    frame.rows * largeurTraitement / frame.cols
                )
            );
        }

        // Conversion en niveaux de gris

        cvtColor(
            frame,
            gris,
            COLOR_BGR2GRAY
        );

        // Traitement : tracking et detection

        double tSuivi = 0;
        double tDetection = 0;

        traiterImage(
            frame,
            gris,
            tSuivi,
            tDetection
        );

        // Recuperer les resultats

        vector<ResultatDetection> resultats =
            recupererResultats();

        // Afficher les cibles detectees

        int nbVisibles = 0;

        for (size_t i = 0; i < cibles.size(); i++)
        {
            if (cibles[i].visible)
            {
                dessinerCible(
                    frame,
                    cibles[i]
                );

                nbVisibles++;
            }
        }

        // Afficher les informations

        putText(
            frame,
            "Images detectees : " +
            to_string(nbVisibles) +
            " / " + to_string(cibles.size()),
            Point(10, 25),
            FONT_HERSHEY_SIMPLEX,
            0.6,
            Scalar(0, 0, 255),
            2
        );

        putText(
            frame,
            "Tracking : " + to_string((int)tSuivi) + " ms",
            Point(10, 50),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar(255, 0, 0),
            2
        );

        putText(
            frame,
            "Detection : " + to_string((int)tDetection) + " ms",
            Point(10, 75),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar(255, 0, 0),
            2
        );

        imshow(
            "Detection et Tracking Multi-Images",
            frame
        );

        // Memoriser la frame precedente

        prevGris = gris.clone();

        imageCourante++;

        // ECHAP pour quitter

        if (waitKey(1) == 27)
            break;
    }

    cap.release();

    destroyAllWindows();

    return 0;
}
