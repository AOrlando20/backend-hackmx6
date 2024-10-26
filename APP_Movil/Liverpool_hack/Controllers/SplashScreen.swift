//
//  ViewController.swift
//  Liverpool_hack
//
//  Created by Alexis Chávez on 25/10/24.
//

import UIKit

class SplashScreen: UIViewController {
    @IBOutlet weak var text1: UILabel!
    @IBOutlet weak var text2: UILabel!
    @IBOutlet weak var text3: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupTextFonts()
        setupBackground()
        addTapGestureRecognizer()
    }
    func setupTextFonts() {
        text1.font = UIFont(name: "Avenir-Heavy", size: 40)
        text1.textColor = UIColor.white
        text2.font = UIFont(name: "Avenir-Medium", size: 25)
        text2.textColor = UIColor.white
        text3.font = UIFont(name: "Avenir", size: 16)
        text3.textColor = UIColor.white
    }
    func addTapGestureRecognizer() {
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(navigateToDashboard))
        view.addGestureRecognizer(tapGesture)
    }
    func setupBackground() {
        let gradientLayer = CAGradientLayer()
        gradientLayer.frame = view.bounds
        gradientLayer.colors = [
            UIColor(red: 1.0, green: 0.5, blue: 0.8, alpha: 1.0).cgColor,
            UIColor(red: 1.0, green: 0.2, blue: 0.6, alpha: 1.0).cgColor
        ]
        gradientLayer.startPoint = CGPoint(x: 0.0, y: 0.0)
        gradientLayer.endPoint = CGPoint(x: 1.0, y: 1.0)
        
        view.layer.insertSublayer(gradientLayer, at: 0)
    }
    
    @objc func navigateToDashboard() {
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        if let dashController = storyboard.instantiateViewController(withIdentifier: "Home") as? Home {
            dashController.modalPresentationStyle = .fullScreen
            self.present(dashController, animated: false, completion: nil)
        }
    }
}

