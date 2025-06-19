# auth/group_admin.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from .admin import COLORS, load_user_tags
from .group_management import GroupService

def create_group_management_layout():
    """Create a group management section for the admin dashboard"""
    group_service = GroupService()
    
    return dbc.Card([
        dbc.CardBody([
            # Groups Management Header
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-users-cog", style={'color': COLORS['primary'], 'marginRight': '10px'}),
                        html.H3("Group Management", style={
                            'color': COLORS['text'],
                            'fontWeight': '600',
                            'margin': '0',
                        })
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], width=8),
                dbc.Col([
                    dbc.Button(
                        ["Add Group ", html.I(className="fas fa-plus")],
                        id="add-group-button",
                        color="primary",
                        n_clicks=0,
                        style={
                            'backgroundColor': COLORS['primary'],
                            'borderColor': COLORS['primary'],
                            'boxShadow': COLORS['shadow'],
                            'fontWeight': '500',
                            'float': 'right'
                        }
                    )
                ], width=4, className="d-flex align-items-center justify-content-end")
            ], className="mb-4"),
            
            # Groups Table
            dash_table.DataTable(
                id='groups-table',
                columns=[
                    {"name": "Group ID", "id": "group_id"},
                    {"name": "Group Name", "id": "name"},
                    {"name": "Description", "id": "description"},
                    {"name": "Group Admins", "id": "group_admins"},
                    {"name": "Members", "id": "members_count"},
                    {"name": "Created At", "id": "created_at"},
                ],
                data=[],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '12px',
                    'whiteSpace': 'normal',
                    'fontFamily': '"Segoe UI", Arial, sans-serif',
                    'fontSize': '14px',
                },
                style_header={
                    'backgroundColor': COLORS['light_accent'],
                    'fontWeight': 'bold',
                    'color': COLORS['text'],
                    'borderBottom': f'1px solid {COLORS["secondary"]}',
                },
                row_selectable="single",
                selected_rows=[],
                page_size=10,
            ),
            
            # Group Action Buttons
            dbc.ButtonGroup([
                dbc.Button(
                    ["Edit Group ", html.I(className="fas fa-edit")],
                    id="edit-group-button",
                    color="warning",
                    className="me-2",
                    disabled=True,
                    style={
                        'fontWeight': '500',
                        'boxShadow': COLORS['shadow'],
                    }
                ),
                dbc.Button(
                    ["Delete Group ", html.I(className="fas fa-trash")],
                    id="delete-group-button",
                    color="danger",
                    disabled=True,
                    style={
                        'fontWeight': '500',
                        'boxShadow': COLORS['shadow'],
                    }
                )
            ], className="mt-3"),
            
            # Add/Edit Group Modal
            dbc.Modal([
                dbc.ModalHeader(
                    dbc.ModalTitle(
                        html.Div(id="group-modal-title"),
                    ),
                    style={'backgroundColor': COLORS['background'], 'borderBottom': f'1px solid {COLORS["light_accent"]}'}
                ),
                dbc.ModalBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Group Name", style={'fontWeight': '500', 'color': COLORS['text']}),
                                dbc.InputGroup([
                                    dbc.InputGroupText(
                                        html.I(className="fas fa-users"),
                                        style={
                                            'backgroundColor': COLORS['background'],
                                            'borderColor': COLORS['light_accent'],
                                            'color': COLORS['primary']
                                        }
                                    ),
                                    dbc.Input(
                                        type="text",
                                        id="group-name-input",
                                        placeholder="Enter group name",
                                        style={
                                            'borderColor': COLORS['light_accent'],
                                            'fontSize': '0.95rem',
                                            'padding': '0.75rem 1rem'
                                        }
                                    )
                                ])
                            ]),
                            dbc.Col([
                                dbc.Label("Description (Optional)", style={'fontWeight': '500', 'color': COLORS['text']}),
                                dbc.InputGroup([
                                    dbc.InputGroupText(
                                        html.I(className="fas fa-comment-alt"),
                                        style={
                                            'backgroundColor': COLORS['background'],
                                            'borderColor': COLORS['light_accent'],
                                            'color': COLORS['primary']
                                        }
                                    ),
                                    dbc.Input(
                                        type="text",
                                        id="group-description-input",
                                        placeholder="Enter group description",
                                        style={
                                            'borderColor': COLORS['light_accent'],
                                            'fontSize': '0.95rem',
                                            'padding': '0.75rem 1rem'
                                        }
                                    )
                                ])
                            ])
                        ])
                    ])
                ], style={'padding': '20px'}),
                dbc.ModalFooter([
                    dbc.Button(
                        "Close",
                        id="close-group-modal-button",
                        className="me-2",
                        n_clicks=0,
                        style={
                            'fontWeight': '500',
                            'boxShadow': COLORS['shadow'],
                        }
                    ),
                    dbc.Button(
                        "Save",
                        id="save-group-button",
                        color="primary",
                        n_clicks=0,
                        style={
                            'backgroundColor': COLORS['primary'],
                            'borderColor': COLORS['primary'],
                            'fontWeight': '500',
                            'boxShadow': COLORS['shadow'],
                        }
                    ),
                ], style={'borderTop': f'1px solid {COLORS["light_accent"]}'}),
            ], id="group-modal", is_open=False, style={'borderRadius': '12px'}),
            
            # Group Members Modal
            dbc.Modal([
                dbc.ModalHeader(
                    dbc.ModalTitle(
                        html.Div(id="group-members-modal-title"),
                    ),
                    style={'backgroundColor': COLORS['background'], 'borderBottom': f'1px solid {COLORS["light_accent"]}'}
                ),
                dbc.ModalBody([
                    dbc.Label("Select Users", style={'fontWeight': '500', 'color': COLORS['text']}),
                    dcc.Dropdown(
                        id='group-members-dropdown',
                        multi=True,
                        placeholder='Select users to add to group',
                        style={
                            'color': COLORS['text'],
                            'borderColor': COLORS['light_accent']
                        }
                    ),
                    
                    # Group Admin Selection 
                    html.Div([
                        dbc.Label("Select Group Admins", style={'fontWeight': '500', 'color': COLORS['text'], 'marginTop': '15px'}),
                        dcc.Dropdown(
                            id='group-admins-dropdown',
                            multi=True,
                            placeholder='Select group admins',
                            style={
                                'color': COLORS['text'],
                                'borderColor': COLORS['light_accent']
                            }
                        )
                    ])
                ], style={'padding': '20px'}),
                dbc.ModalFooter([
                    dbc.Button(
                        "Close",
                        id="close-group-members-modal-button",
                        className="me-2",
                        n_clicks=0,
                        style={
                            'fontWeight': '500',
                            'boxShadow': COLORS['shadow'],
                        }
                    ),
                    dbc.Button(
                        "Save Members",
                        id="save-group-members-button",
                        color="primary",
                        n_clicks=0,
                        style={
                            'backgroundColor': COLORS['primary'],
                            'borderColor': COLORS['primary'],
                            'fontWeight': '500',
                            'boxShadow': COLORS['shadow'],
                        }
                    ),
                ], style={'borderTop': f'1px solid {COLORS["light_accent"]}'}),
            ], id="group-members-modal", is_open=False, style={'borderRadius': '12px'}),
            
            # Feedback Toast
            dbc.Toast(
                id="group-feedback-toast",
                header="Group Notification",
                is_open=False,
                dismissable=True,
                duration=4000,
                style={
                    "position": "fixed", 
                    "top": 66, 
                    "right": 10, 
                    "width": 350, 
                    "boxShadow": COLORS['shadow']
                },
            ),
        ])
    ], style={
        'borderRadius': '12px',
        'border': f'1px solid {COLORS["light_accent"]}',
        'boxShadow': COLORS['shadow'],
        'marginTop': '1.5rem'
    })

# Additional helper functions (if needed)
def prepare_group_modal_data(group_service, db_session):
    """
    Prepare data for group modal dropdowns
    
    :param group_service: GroupService instance
    :param db_session: Database session
    :return: Dictionary with user options
    """
    # Fetch all users for dropdown
    users = db_session.query(User).all()
    user_options = [
        {"label": user.username, "value": str(user.id)} 
        for user in users
    ]
    
    return {
        "user_options": user_options
    }