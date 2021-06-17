#pragma once
//#include <wx/wxprec.h>

class MainApp :public wxApp {
public:
	virtual bool OnInit();
};


class MainFrame :public wxFrame {
public:
	MainFrame(const wxString& title, const wxPoint& pos, const wxSize& size);
	//wxButton* HelloWorld;
	wxTextCtrl* MainEditBox;
	wxMenuBar* MainMenu;
	void Quit(wxCommandEvent& event);
	void NewFile(wxCommandEvent& event);
	void OpenFile(wxCommandEvent& event);
	void SaveFile(wxCommandEvent& event);
	void SaveFileAs(wxCommandEvent& event);
	void CloseFile(wxCommandEvent& event);
	void OnExit(wxCommandEvent& event);

	DECLARE_EVENT_TABLE()
};

enum {
	//BUTTON_Hello = wxID_HIGHEST + 1
	TEXT_Main = wxID_HIGHEST + 1,
	MENU_New,
	MENU_Open,
	MENU_Close,
	MENU_Save,
	MENU_SaveAs,
	MENU_Quit
};